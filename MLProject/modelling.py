import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Heart_Disease_Basic")
mlflow.sklearn.autolog()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_split_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).iloc[:, 0]
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).iloc[:, 0]

    return X_train, X_test, y_train, y_test

def train_basic(X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("Accuracy (Basic Model):", acc)
        
        os.makedirs("model", exist_ok=True)
        mlflow.sklearn.save_model(model, "model")

def main():
    data_dir = os.path.join(BASE_DIR, "heart_preprocessing")
    X_train, X_test, y_train, y_test = load_split_data(data_dir)
    train_basic(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()


