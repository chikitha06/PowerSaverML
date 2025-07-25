import pandas as pd 
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 1: Generate simulated dataset with improved overheating logic
data = []
for _ in range(1000):
    app_type = random.choice(["Game", "Video", "Social Media", "Productivity"])
    charging_status = random.choice(["Yes", "No"])

    # Generate CPU usage based on app type and charging status
    if app_type == "Game" and charging_status == "Yes":
        cpu_usage = random.randint(90, 100)
    elif app_type == "Video":
        cpu_usage = random.randint(70, 90)
    elif app_type == "Social Media":
        cpu_usage = random.randint(60, 80)
    elif app_type == "Productivity":
        cpu_usage = random.randint(30, 60)
    else:
        cpu_usage = random.randint(10, 25)

    screen_brightness = random.randint(50, 100)
    battery_level = random.randint(10, 100)

    # Label overheating based on multiple realistic combinations
    if (cpu_usage > 85 and screen_brightness > 80 and app_type == "Game") or \
       (cpu_usage > 90 and charging_status == "Yes") or \
       (cpu_usage > 80 and screen_brightness > 85 and app_type in ["Game", "Video"]) or \
       (charging_status == "Yes" and screen_brightness > 90 and app_type == "Video"):
        is_overheating = 1
    else:
        is_overheating = 0

    data.append([cpu_usage, battery_level, screen_brightness, app_type, charging_status, is_overheating])

# Step 2: Create a DataFrame
df = pd.DataFrame(data, columns=[
    "cpu_usage", "battery_level", "screen_brightness",
    "app_type", "charging_status", "is_overheating"
])

# Optional: Save the dataset
df.to_csv("overheating_data.csv", index=False)

# Step 3: One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=["app_type", "charging_status"], drop_first=True)

# Step 4: Split into features and target
X = df_encoded.drop("is_overheating", axis=1)
y = df_encoded["is_overheating"]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 6: Train the Random Forest classifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%\n")

print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


