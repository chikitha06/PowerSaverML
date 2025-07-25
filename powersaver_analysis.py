# eda_overheating.py

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Step 1: Load dataset
df = pd.read_csv("C:\Users\Atava\OneDrive\Desktop\gitdemo\PowerSaverML\overheating_data.csv")

# Step 2: Summary
print("📄 Dataset Info:\n")
print(df.info())
print("\n🔢 Statistical Summary:\n")
print(df.describe())

# Step 3: Target distribution
target_counts = df["is_overheating"].value_counts().sort_index()
labels = ["Not Overheating", "Overheating"]
colors = ["lightblue", "salmon"]

target_counts.plot(kind="bar", color=colors)
plt.xticks([0, 1], labels, rotation=0)
plt.title("Overheating Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# Step 4: Compare CPU Usage across overheating classes
plt.hist(
    [df[df["is_overheating"] == 0]["cpu_usage"],
     df[df["is_overheating"] == 1]["cpu_usage"]],
    bins=20,
    stacked=True,
    color=colors,
    label=labels,
    edgecolor="black"
)
plt.title("CPU Usage vs Overheating")
plt.xlabel("CPU Usage (%)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Step 5: Compare Screen Brightness across classes
plt.hist(
    [df[df["is_overheating"] == 0]["screen_brightness"],
     df[df["is_overheating"] == 1]["screen_brightness"]],
    bins=20,
    stacked=True,
    color=colors,
    label=labels,
    edgecolor="black"
)
plt.title("Screen Brightness vs Overheating")
plt.xlabel("Brightness (%)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Step 6: Categorical comparison – App Type
app_counts = df.groupby("app_type")["is_overheating"].value_counts().unstack().fillna(0)
app_counts.columns = labels
app_counts.plot(kind="bar", color=colors)
plt.title("App Type vs Overheating")
plt.xlabel("App Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 7: Categorical comparison – Charging Status
charge_counts = df.groupby("charging_status")["is_overheating"].value_counts().unstack().fillna(0)
charge_counts.columns = labels
charge_counts.plot(kind="bar", color=colors)
plt.title("Charging Status vs Overheating")
plt.xlabel("Charging")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Step 8: Correlation heatmap (manual)
df_corr = df.copy()
df_corr["charging_status"] = df_corr["charging_status"].map({"Yes": 1, "No": 0})
df_corr["app_type"] = df_corr["app_type"].astype("category").cat.codes

correlation_matrix = df_corr.corr()
print("\n📈 Correlation Matrix:\n", correlation_matrix)

fig, ax = plt.subplots()
cax = ax.matshow(correlation_matrix, cmap="coolwarm")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Heatmap", pad=20)
plt.colorbar(cax)
plt.tight_layout()
plt.show()
