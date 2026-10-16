import os
import getpass
import pandas as pd
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "student_db")
DB_USER = os.getenv("DB_USER") or getpass.getuser()
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

LOG_FILE = os.path.join(BASE_DIR, "logs", "drift_and_retraining.log")

def check_feature_drift():
    conn_params = {"host": DB_HOST, "port": DB_PORT, "dbname": DB_NAME, "user": DB_USER}
    if DB_PASSWORD:
        conn_params["password"] = DB_PASSWORD
        
    conn = psycopg2.connect(**conn_params)
    df = pd.read_sql_query('SELECT "attendance_pct", "avg_internal_marks", "risk_level" FROM "fact_student_analytics";', conn)
    conn.close()
    
    expected_avg_attendance = 90.0
    expected_risk_ratio = 0.35
    
    current_avg_attendance = df['attendance_pct'].mean()
    current_risk_ratio = (df['risk_level'] == 'High Risk').mean()
    
    attendance_shift = abs(current_avg_attendance - expected_avg_attendance)
    risk_shift = abs(current_risk_ratio - expected_risk_ratio)
    
    retrain_needed = bool(attendance_shift > 5.0 or risk_shift > 0.10)
    
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"\n--- MONITORING RUN: {datetime.now().isoformat()} ---\n")
        f.write(f"Current Avg Attendance: {current_avg_attendance:.2f}% (Expected: {expected_avg_attendance}%)\n")
        f.write(f"Current Risk Proportion: {current_risk_ratio*100:.2f}% (Expected: {expected_risk_ratio*100}%)\n")
        f.write(f"Retraining Required: {retrain_needed}\n")
        
    print(f"Monitoring complete. Retraining Required: {retrain_needed}")
    return retrain_needed

if __name__ == "__main__":
    check_feature_drift()