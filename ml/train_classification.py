import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURE_PATH = r"C:\stockmarket\data_features"
MODEL_PATH = r"C:\stockmarket\models"

os.makedirs(MODEL_PATH, exist_ok=True)

FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume','daily_return', 'sma_10', 'sma_50']

for file in os.listdir(FEATURE_PATH):
    if file.endswith(".csv"):
        print(f"\nTraining Classification Model for: {file}")

        df = pd.read_csv(os.path.join(FEATURE_PATH, file))

        X = df[FEATURE_COLUMNS]
        y = df['target_trend']

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, shuffle=False)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        model_name = file.replace(".csv", "_rf_classification.pkl")
        joblib.dump(model, os.path.join(MODEL_PATH, model_name))

        print(" Classification Model Saved:", model_name)
