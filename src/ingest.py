import os
import pandas as pd
from datetime import datetime

# Resolve absolute path to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_FILE = os.path.join(RAW_DIR, "raw_student_math.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
UCI_URL = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv"

def run_ingestion():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"[{datetime.now()}] Ingesting data from UCI repository...")
    df_raw = pd.read_csv(UCI_URL)
    df_raw.to_csv(RAW_FILE, index=False)
    
    audit_log = (
        f"TIMESTAMP: {datetime.now().isoformat()} | "
        f"SOURCE: UCI Student Performance | "
        f"STATUS: SUCCESS | "
        f"ROWS: {len(df_raw)} | "
        f"SAVED: {RAW_FILE}\n"
    )
    with open(os.path.join(LOG_DIR, "extraction_audit.log"), "a") as f:
        f.write(audit_log)
        
    print(f"Raw ingestion complete: {len(df_raw)} records stored.")
    return RAW_FILE

if __name__ == "__main__":
    run_ingestion()