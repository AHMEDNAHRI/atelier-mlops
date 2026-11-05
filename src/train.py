import os
import json
import yaml
import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]

df = pd.read_csv("data/raw/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y,
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def run_model(run_name, model, model_params, model_path, metrics_path):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted"),
            "recall": recall_score(y_test, predictions, average="weighted"),
            "f1_score": f1_score(y_test, predictions, average="weighted"),
        }

        mlflow.log_param("model_type", model.__class__.__name__)
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", params["test_size"])
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_metrics(metrics)

        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(run_name, "->", metrics)


n_est = params["n_estimators"]
depth = params["max_depth"]
k = params["n_neighbors"]
seed = params["random_state"]

run_model(
    "RandomForest_n" + str(n_est) + "_d" + str(depth),
    RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=seed),
    {"n_estimators": n_est, "max_depth": depth},
    "models/model.pkl",
    "reports/metrics.json",
)

run_model(
    "KNN_k" + str(k),
    KNeighborsClassifier(n_neighbors=k),
    {"n_neighbors": k},
    "models/knn_model.pkl",
    "reports/metrics_knn.json",
)

print("Entrainement termine.")
