import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error


# Load dataset
df = pd.read_csv("data/synthetic_data.csv")


# Create interaction terms
df["AB"] = df["dispersion_A"] * df["dispersion_B"]
df["AC"] = df["dispersion_A"] * df["dispersion_C"]
df["BC"] = df["dispersion_B"] * df["dispersion_C"]


# Input features
X = df[
    [
        "dispersion_A",
        "dispersion_B",
        "dispersion_C",
        "AB",
        "AC",
        "BC",
    ]
]

# Targets
y = df[["L", "a", "b"]]


# Train quadratic mixture model
model = LinearRegression()
model.fit(X, y)


# Predict training data
y_pred = model.predict(X)


# Evaluate
r2 = r2_score(y, y_pred)
rmse = root_mean_squared_error(y, y_pred)

print("Quadratic Mixture Model")
print("-----------------------")
print(f"R²:   {r2:.3f}")
print(f"RMSE: {rmse:.3f}")