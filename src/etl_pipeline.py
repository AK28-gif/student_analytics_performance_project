import os
import getpass
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "student_db")
DB_USER = os.getenv("DB_USER") or getpass.getuser()
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

REJECTED_LOG = os.path.join(BASE_DIR, "logs", "error_and_rejected_records.log")

def write_df_to_postgres(df, table_name, conn):
    cur = conn.cursor()
    type_map = {
        'int64': 'INTEGER',
        'float64': 'DOUBLE PRECISION',
        'object': 'TEXT',
        'bool': 'BOOLEAN'
    }
    
    col_defs = []
    for col_name, dtype in df.dtypes.items():
        pg_type = type_map.get(str(dtype), 'TEXT')
        col_defs.append(f'"{col_name}" {pg_type}')
    
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    cur.execute(f'CREATE TABLE "{table_name}" ({", ".join(col_defs)});')
    
    columns = [f'"{col}"' for col in df.columns]
    insert_query = f'INSERT INTO "{table_name}" ({", ".join(columns)}) VALUES %s'
    values = [tuple(x) for x in df.to_numpy()]
    
    execute_values(cur, insert_query, values)
    conn.commit()
    cur.close()

def run_etl():
    raw_path = os.path.join(BASE_DIR, "data", "raw", "raw_student_math.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Run ingestion first.")

    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} records for transformation.")

    # 1. Standardize Student Identifiers
    df['student_id'] = [f"STU_{i+1001}" for i in range(len(df))]

    # 2. Quality Checks & Error Logging
    rejected = []
    invalid_grades = df[(df['G1'] < 0) | (df['G1'] > 20) | 
                        (df['G2'] < 0) | (df['G2'] > 20) | 
                        (df['G3'] < 0) | (df['G3'] > 20)]
    for _, row in invalid_grades.iterrows():
        rejected.append(f"Student {row['student_id']}: Invalid assessment grade outside 0-20.")

    invalid_absences = df[df['absences'] < 0]
    for _, row in invalid_absences.iterrows():
        rejected.append(f"Student {row['student_id']}: Invalid negative absences.")

    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    with open(REJECTED_LOG, "w") as f:
        f.write(f"ETL Validation Audit - {datetime.now().isoformat()}\n")
        f.write(f"Total anomalies flagged: {len(rejected)}\n")
        for line in rejected:
            f.write(line + "\n")

    # Clean duplicates and valid boundaries
    df_clean = df.drop_duplicates(subset=['student_id']).copy()
    df_clean = df_clean[(df_clean['G3'] >= 0) & (df_clean['G3'] <= 20)]

    # 3. Feature Engineering
    total_sessions = 93
    df_clean['attendance_pct'] = np.round(((total_sessions - df_clean['absences']) / total_sessions) * 100, 2)
    df_clean['attendance_pct'] = df_clean['attendance_pct'].clip(0, 100)
    df_clean['avg_internal_marks'] = np.round((df_clean['G1'] + df_clean['G2']) / 2.0, 2)
    df_clean['final_grade'] = df_clean['G3']
    df_clean['pass_status'] = np.where(df_clean['final_grade'] >= 10, 'Pass', 'Fail')
    df_clean['risk_level'] = np.where(
        (df_clean['final_grade'] < 10) | (df_clean['attendance_pct'] < 75.0),
        'High Risk',
        'Normal'
    )

    # 4. Load into PostgreSQL
    print(f"Connecting to PostgreSQL (db: {DB_NAME}, user: {DB_USER})...")
    conn_params = {
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER
    }
    if DB_PASSWORD:
        conn_params["password"] = DB_PASSWORD

    conn = psycopg2.connect(**conn_params)
    try:
        write_df_to_postgres(df_clean, "stg_student_records", conn)
        dim_student = df_clean[['student_id', 'school', 'sex', 'age', 'address', 'famsize', 'Medu', 'Fedu', 'studytime']]
        write_df_to_postgres(dim_student, "dim_student", conn)
        fact_student_analytics = df_clean[[
            'student_id', 'school', 'studytime', 'failures', 'absences',
            'attendance_pct', 'avg_internal_marks', 'final_grade', 'pass_status', 'risk_level'
        ]]
        write_df_to_postgres(fact_student_analytics, "fact_student_analytics", conn)
        print("PostgreSQL loading successful.")
    finally:
        conn.close()

if __name__ == "__main__":
    run_etl()