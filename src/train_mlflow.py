import os
import getpass
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "student_db")
DB_USER = os.getenv("DB_USER") or getpass.getuser()
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

def fetch_training_data():
    conn_params = {"host": DB_HOST, "port": DB_PORT, "dbname": DB_NAME, "user": DB_USER}
    if DB_PASSWORD:
        conn_params["password"] = DB_PASSWORD
        
    conn = psycopg2.connect(**conn_params)
    query = """
        SELECT f."studytime", f."failures", f."absences", f."attendance_pct", f."avg_internal_marks",
               CASE WHEN f."risk_level" = 'High Risk' THEN 1 ELSE 0 END AS label
        FROM "fact_student_analytics" f;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def run_ml_pipeline():
    df = fetch_training_data()
    X = df[["studytime", "failures", "absences", "attendance_pct", "avg_internal_marks"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cleanly end any active runs
    if mlflow.active_run():
        mlflow.end_run()

    # Use file store tracking to prevent SQLite concurrency deadlocks in Airflow
    mlflow_tracking_dir = os.path.join(BASE_DIR, "mlruns")
    os.makedirs(mlflow_tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
    mlflow.set_experiment("Student_Dropout_Risk_Prediction")

    models = {
        "Logistic_Regression": (LogisticRegression(random_state=42, max_iter=1000), True),
        "Random_Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False),
        "Gradient_Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=42), False)
    }

    best_model = None
    best_f1 = -1.0
    best_model_name = ""

    for name, (model, use_scaled) in models.items():
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=name):
            xtrain = X_train_scaled if use_scaled else X_train
            xtest = X_test_scaled if use_scaled else X_test

            model.fit(xtrain, y_train)
            preds = model.predict(xtest)
            probs = model.predict_proba(xtest)[:, 1]

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            roc_auc = roc_auc_score(y_test, probs)

            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"[{name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model = model

    if mlflow.active_run():
        mlflow.end_run()

    print(f"\nWinning Production Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    return best_model_name

if __name__ == "__main__":
    run_ml_pipeline()