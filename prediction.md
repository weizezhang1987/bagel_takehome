# Prediction

Date: 2026-04-24

## Scaling Law Hypothesis

For CIFAR-10 width-scaled CNN models, validation error E(N) decreases with parameter count N following:

E(N) = aN^-b + c

where:
- a controls scale
- b is scaling exponent
- c is irreducible error floor

## Expected Behavior

As model width increases, representation power improves and validation error decreases.

Returns should diminish at larger scales due to:
1. Fixed dataset size
2. Task complexity ceiling
3. Optimization saturation

## Holdout Prediction

The largest held-out model should lie near the extrapolated curve, with modest deviation.