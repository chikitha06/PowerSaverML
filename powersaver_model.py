import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def simulate_data(n=1000):
    data = {
        'cpu_usage': np.random.randint(1, 100, size=n),
        'battery_level': np.random.randint(1, 100, size=n),
        'screen_brightness': np.random.randint(0, 101, size=n),
        'app_type': np.random.choice(['Game', 'Social', 'Streaming', 'Utility'], size=n),
        'charging': np.random.choice(['Yes', 'No'], size=n),
    }

    df = pd.DataFrame(data)

    def overheating(row):
        if row['cpu_usage'] > 70 and row['screen_brightness'] > 70 and row['charging'] == 'Yes':
            return 1
        elif row['cpu_usage'] > 85 and row['battery_level'] < 20:
            return 1
        else:
            return 0

    df['overheating'] = df.apply(overheating, axis=1)
    return df

df = simulate_data()
print("Sample Data:\n", df.head())

X = df.drop('overheating', axis=1)
y = df['overheating']

X = pd.get_dummies(X, columns=['app_type', 'charging'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_overheating():
    print("\n--- Real-Time Overheating Predictor ---")
    cpu = int(input("Enter CPU usage (1-100): "))
    battery = int(input("Enter Battery level (1-100): "))
    brightness = int(input("Enter Screen brightness (0-100): "))
    app = input("Enter App type (Game/Social/Streaming/Utility): ")
    charge = input("Is phone charging? (Yes/No): ")

    input_dict = {
        'cpu_usage': [cpu],
        'battery_level': [battery],
        'screen_brightness': [brightness],
        'app_type_Game': [1 if app == 'Game' else 0],
        'app_type_Social': [1 if app == 'Social' else 0],
        'app_type_Streaming': [1 if app == 'Streaming' else 0],
        'app_type_Utility': [1 if app == 'Utility' else 0],
        'charging_Yes': [1 if charge == 'Yes' else 0],
    }

    input_df = pd.DataFrame(input_dict)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    print("\n🔍 Overheating Prediction:", "⚠️ Yes" if prediction else "✅ No")
    print("🔥 Overheating Probability:", round(probability * 100, 2), "%")


overheat_counts = df['overheating'].value_counts()
labels = ['No Overheat', 'Overheat']

plt.bar(labels, overheat_counts)
plt.title("Overheating Occurrence")
plt.ylabel("Number of Samples")
plt.show()
