import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("data/synthetic_data.csv")

y = df[["L", "a", "b"]].values


# -----------------------------
# Feature sets
# -----------------------------

X_linear = df[
    ["dispersion_A", "dispersion_B", "dispersion_C"]
].values


X_quadratic = pd.DataFrame({
    "A": df["dispersion_A"],
    "B": df["dispersion_B"],
    "C": df["dispersion_C"],
    "AB": df["dispersion_A"] * df["dispersion_B"],
    "AC": df["dispersion_A"] * df["dispersion_C"],
    "BC": df["dispersion_B"] * df["dispersion_C"],
}).values


# -----------------------------
# LOOCV
# -----------------------------

def evaluate_loocv(X, y):

    loo = LeaveOneOut()

    predictions = np.zeros_like(y, dtype=float)

    for train_index, test_index in loo.split(X):

        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions[test_index] = model.predict(X_test)

    # ΔE*ab (CIE76)
    delta_e = np.sqrt(
        np.sum((y - predictions) ** 2, axis=1)
    )

    return predictions, delta_e


# -----------------------------
# Evaluate models
# -----------------------------

_, de_linear = evaluate_loocv(X_linear, y)
_, de_quadratic = evaluate_loocv(X_quadratic, y)


# -----------------------------
# Results
# -----------------------------

print("LOOCV Results")
print("-------------")

print("\nLinear Model")
print(f"Mean ΔE:   {np.mean(de_linear):.3f}")
print(f"Median ΔE: {np.median(de_linear):.3f}")
print(f"Max ΔE:    {np.max(de_linear):.3f}")

print("\nQuadratic Mixture Model")
print(f"Mean ΔE:   {np.mean(de_quadratic):.3f}")
print(f"Median ΔE: {np.median(de_quadratic):.3f}")
print(f"Max ΔE:    {np.max(de_quadratic):.3f}")


results = pd.DataFrame({
    "sample": df["sample"],
    "linear_deltaE": de_linear,
    "quadratic_deltaE": de_quadratic
})

print("\nLargest Quadratic Errors")
print(
    results
    .sort_values("quadratic_deltaE", ascending=False)
    .head(5)
)

# Separate pure-component endpoints from interior mixtures
pure_mask = (
    (df["dispersion_A"] == 1.0)
    | (df["dispersion_B"] == 1.0)
    | (df["dispersion_C"] == 1.0)
)

print("\nInterior Mixtures Only")
print("----------------------")

print("\nLinear Model")
print(f"Mean ΔE:   {np.mean(de_linear[~pure_mask]):.3f}")
print(f"Median ΔE: {np.median(de_linear[~pure_mask]):.3f}")
print(f"Max ΔE:    {np.max(de_linear[~pure_mask]):.3f}")

print("\nQuadratic Mixture Model")
print(f"Mean ΔE:   {np.mean(de_quadratic[~pure_mask]):.3f}")
print(f"Median ΔE: {np.median(de_quadratic[~pure_mask]):.3f}")
print(f"Max ΔE:    {np.max(de_quadratic[~pure_mask]):.3f}")


# -----------------------------
# Visualize prediction errors
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df["sample"],
    de_linear,
    marker="o",
    label="Linear"
)

plt.plot(
    df["sample"],
    de_quadratic,
    marker="o",
    label="Quadratic mixture"
)

plt.xlabel("Sample")
plt.ylabel("ΔE*ab")
plt.title("LOOCV Prediction Error by Sample")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures/loocv_deltaE_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()