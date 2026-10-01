import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("data/synthetic_data.csv")

y = df[["L", "a", "b"]].values


# -----------------------------
# Quadratic mixture features
# -----------------------------

X = pd.DataFrame({
    "A": df["dispersion_A"],
    "B": df["dispersion_B"],
    "C": df["dispersion_C"],
    "AB": df["dispersion_A"] * df["dispersion_B"],
    "AC": df["dispersion_A"] * df["dispersion_C"],
    "BC": df["dispersion_B"] * df["dispersion_C"],
}).values


# -----------------------------
# Ridge + LOOCV
# -----------------------------

alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]

loo = LeaveOneOut()

# Identify pure-component endpoints
pure_mask = (
    (df["dispersion_A"] == 1.0)
    | (df["dispersion_B"] == 1.0)
    | (df["dispersion_C"] == 1.0)
)

results = []

for alpha in alphas:

    predictions = np.zeros_like(y, dtype=float)

    for train_index, test_index in loo.split(X):

        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        predictions[test_index] = model.predict(X_test)

    # ΔE*ab (CIE76)
    delta_e = np.sqrt(
        np.sum((y - predictions) ** 2, axis=1)
    )

    # Evaluate interior mixtures only
    interior_delta_e = delta_e[~pure_mask]

    results.append({
    "alpha": alpha,
    "Mean DeltaE": np.mean(delta_e),
    "Median DeltaE": np.median(delta_e),
    "Max DeltaE": np.max(delta_e),
    "Interior Mean DeltaE": np.mean(interior_delta_e),
    "Interior Median DeltaE": np.median(interior_delta_e),
    "Interior Max DeltaE": np.max(interior_delta_e),
})


# -----------------------------
# Results
# -----------------------------

results_df = pd.DataFrame(results)

print("Ridge LOOCV Results")
print("-------------------")
print(results_df)


# -----------------------------
# Visualize Ridge performance
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["alpha"],
    results_df["Mean DeltaE"],
    marker="o",
    label="All samples"
)

plt.plot(
    results_df["alpha"],
    results_df["Interior Mean DeltaE"],
    marker="o",
    label="Interior mixtures"
)

plt.xscale("log")

plt.xlabel("Ridge alpha")
plt.ylabel("Mean ΔE*ab")
plt.title("Effect of Ridge Regularization on LOOCV Error")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures/ridge_alpha_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()