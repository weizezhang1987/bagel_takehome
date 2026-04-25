# fit_scaling_law.py
# Fully adapted for current results/results.csv format:
# columns = width,params,epochs,seed,val_loss,val_acc
#
# Behavior:
# 1. Keep only rows with TARGET_EPOCHS
# 2. Deduplicate same width (keep latest)
# 3. Sort by params
# 4. Use first N-1 points to fit scaling law
# 5. Use largest model as holdout
# 6. Plot accuracy scaling curve
# 7. Save figure to results/scaling_law.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =====================================================
# Config
# =====================================================
CSV_PATH = "results/results.csv"
PLOT_DIR = "results"
SAVE_PATH = os.path.join(PLOT_DIR, "scaling_law.png")

TARGET_EPOCHS = 15   # only use final experiments
USE_ERROR_METRIC = True  # fit on error = 1 - acc

# =====================================================
# Helpers
# =====================================================
def scaling_fn(N, a, b, c):
    """
    Power-law:
        metric = a * N^(-b) + c
    """
    return a * np.power(N, -b) + c


# =====================================================
# Load CSV
# =====================================================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Cannot find {CSV_PATH}")

os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

required_cols = ["width", "params", "epochs", "seed", "val_loss", "val_acc"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Keep only target epochs
df = df[df["epochs"] == TARGET_EPOCHS].copy()

if len(df) < 2:
    raise ValueError("Need at least 2 rows after filtering.")

# Keep latest duplicate width
df = df.drop_duplicates(subset=["width"], keep="last")

# Sort by params
df = df.sort_values("params").reset_index(drop=True)

# Convert accuracy -> error if enabled
if USE_ERROR_METRIC:
    df["metric"] = 1.0 - df["val_acc"]
    metric_name = "Error Rate"
else:
    df["metric"] = df["val_acc"]
    metric_name = "Validation Accuracy"

print("=" * 60)
print("Filtered Data")
print("=" * 60)
print(df[["width", "params", "epochs", "val_acc", "metric"]])

# =====================================================
# Train / Holdout split
# =====================================================
train_df = df.iloc[:-1].copy()
test_df = df.iloc[-1:].copy()

x_train = train_df["params"].values.astype(float)
y_train = train_df["metric"].values.astype(float)

x_test = float(test_df["params"].values[0])
y_test = float(test_df["metric"].values[0])

# =====================================================
# Fit curve
# =====================================================
# Initial guess
if USE_ERROR_METRIC:
    p0 = [1.0, 0.15, min(y_train) * 0.8]
else:
    p0 = [0.1, 0.1, max(y_train) * 0.8]

params_opt, _ = curve_fit(
    scaling_fn,
    x_train,
    y_train,
    p0=p0,
    maxfev=20000
)

a, b, c = params_opt

# Prediction
pred_test_metric = scaling_fn(x_test, a, b, c)

# Convert back to accuracy for reporting
if USE_ERROR_METRIC:
    pred_test_acc = 1.0 - pred_test_metric
    true_test_acc = float(test_df["val_acc"].values[0])
else:
    pred_test_acc = pred_test_metric
    true_test_acc = float(test_df["val_acc"].values[0])

abs_err = abs(pred_test_acc - true_test_acc)

# =====================================================
# Print Results
# =====================================================
print("\n" + "=" * 60)
print("Fitted Scaling Law")
print("=" * 60)
print(f"{metric_name} = a * N^(-b) + c")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")

print("\nHoldout Evaluation")
print("-" * 60)
print(f"Holdout width   : {int(test_df['width'].values[0])}")
print(f"Holdout params  : {int(x_test):,}")
print(f"Predicted acc   : {pred_test_acc:.4f}")
print(f"Actual acc      : {true_test_acc:.4f}")
print(f"Absolute error  : {abs_err:.4f}")

# =====================================================
# Plot
# =====================================================
x_plot = np.logspace(
    np.log10(df["params"].min()),
    np.log10(df["params"].max()),
    300
)

y_plot = scaling_fn(x_plot, a, b, c)

# convert metric to accuracy for display
if USE_ERROR_METRIC:
    acc_curve = 1.0 - y_plot
else:
    acc_curve = y_plot

plt.figure(figsize=(10, 6))

# fit points
plt.scatter(
    train_df["params"],
    train_df["val_acc"],
    s=80,
    label="Fit Points"
)

# holdout actual
plt.scatter(
    [x_test],
    [true_test_acc],
    s=120,
    marker="x",
    linewidths=2,
    label="Holdout Actual"
)

# holdout predicted
plt.scatter(
    [x_test],
    [pred_test_acc],
    s=120,
    marker="^",
    label="Holdout Predicted"
)

# fitted curve
plt.plot(
    x_plot,
    acc_curve,
    linewidth=2,
    label="Power-law Fit"
)

plt.xscale("log")
plt.xlabel("Parameter Count (log scale)")
plt.ylabel("Validation Accuracy")
plt.title("CNN Scaling Law on CIFAR-10")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(SAVE_PATH, dpi=220)
plt.show()

print(f"\nSaved plot to: {SAVE_PATH}")