import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

FEATURE_DATA_PATH = r"C:\stockmarket\data_features"
MODEL_PATH = r"C:\stockmarket\models"

os.makedirs(MODEL_PATH, exist_ok=True)

def train_regression_model(file_path):
    df = pd.read_csv(file_path)

    X = df[
        [
            'open',
            'high',
            'low',
            'close',
            'volume',
            'daily_return',
            'sma_10',
            'sma_50'
        ]
    ]

    y = df['target_price']

    # Time-series split (NO SHUFFLE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Regression Results")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("R2  :", r2_score(y_test, y_pred))

    return model


if __name__ == "__main__":
    for file in os.listdir(FEATURE_DATA_PATH):
        if file.endswith(".csv"):
            path = os.path.join(FEATURE_DATA_PATH, file)
            print("\nTraining model for:", file)

            model = train_regression_model(path)

            model_name = file.replace(".csv", "_regression.pkl")
            save_path = os.path.join(MODEL_PATH, model_name)

            joblib.dump(model, save_path)
            print("Model saved:", save_path)
