import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/overheating_data.csv") 
print("Dataset shape:", df.shape)

label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target
X = df.drop("is_overheating", axis=1)
y = df["is_overheating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training size: {len(X_train)}, Testing size: {len(X_test)}")

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest
print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:\n", cm_rf)
ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot()

sample = X_test.iloc[0]
sample_df = pd.DataFrame([sample], columns=X_test.columns)
sample_prediction = rf.predict(sample_df)[0]

print("\n--- Prediction Result ---")
if sample_prediction == 1:
    print("⚠️ Warning: Your smartphone is likely to overheat. Consider closing some apps or reducing brightness.")
else:
    print("✅ All good! Your phone is operating within safe thermal limits.")



