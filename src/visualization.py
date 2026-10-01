import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load synthetic dataset
df = pd.read_csv("data/synthetic_data.csv")

print(df.head())
print(df.shape)


# -----------------------------
# Visualize color distribution
# -----------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    df["a"],
    df["b"]
)

plt.xlabel("a*")
plt.ylabel("b*")
plt.title("Color Distribution in CIELAB Space")

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures/ab_color_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# Visualize colors in 3D CIELAB space
# -----------------------------

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["a"],
    df["b"],
    df["L"]
)

ax.set_xlabel("a*")
ax.set_ylabel("b*")
ax.set_zlabel("L*")
ax.set_title("Color Distribution in CIELAB Space")

plt.tight_layout()

plt.savefig(
    "figures/cielab_3d_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()