import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load synthetic dataset
df = pd.read_csv("data/synthetic_data.csv")

print(df.head())
print(df.shape)

# -----------------------------
# Data validation
# -----------------------------

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check that mixture fractions sum to 1
df["mixture_sum"] = (
    df["dispersion_A"]
    + df["dispersion_B"]
    + df["dispersion_C"]
)

print("\nMixture sum:")
print(df["mixture_sum"])

print(
    "\nAll mixtures sum to 1:",
    np.allclose(df["mixture_sum"], 1.0)
)