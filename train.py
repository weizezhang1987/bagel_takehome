import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self, width):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(3,         width),
            block(width,     width),
            nn.MaxPool2d(2),                    # 32 -> 16

            block(width,     width * 2),
            block(width * 2, width * 2),
            nn.MaxPool2d(2),                    # 16 -> 8

            block(width * 2, width * 4),
            block(width * 4, width * 4),
            nn.AdaptiveAvgPool2d(1),            # 8 -> 1
        )
        self.classifier = nn.Linear(width * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_loaders(batch_size):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10("data", train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=val_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def save_result(width, params, epochs, seed, val_loss, val_acc):
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "results.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["width", "params", "epochs", "seed", "val_loss", "val_acc"])
        writer.writerow([width, params, epochs, seed, f"{val_loss:.6f}", f"{val_acc:.4f}"])


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 scaling law experiment")
    parser.add_argument("--width",      type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  AMP: {use_amp}")

    model = CNN(args.width).to(device)
    params = count_params(model)
    print(f"Width: {args.width}  Params: {params:,}")

    train_loader, val_loader = get_loaders(args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss           = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc    = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_acc:>7.4f}  {elapsed:>5.1f}s")

    save_result(args.width, params, args.epochs, args.seed, val_loss, val_acc)
    print(f"\nSaved to results/results.csv  (width={args.width}, params={params:,}, val_acc={val_acc:.4f})")


if __name__ == "__main__":
    main()
