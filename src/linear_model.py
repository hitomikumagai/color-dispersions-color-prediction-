import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error


# Load dataset
df = pd.read_csv("data/synthetic_data.csv")


# Input features: mixture composition
X = df[
    ["dispersion_A", "dispersion_B", "dispersion_C"]
]

# Targets: CIELAB values
y = df[["L", "a", "b"]]


# Train linear regression model
model = LinearRegression()
model.fit(X, y)


# Predict the training data
y_pred = model.predict(X)


# Evaluate model
r2 = r2_score(y, y_pred)
rmse = root_mean_squared_error(y, y_pred)

print("Linear Regression")
print("-----------------")
print(f"R²:   {r2:.3f}")
print(f"RMSE: {rmse:.3f}")