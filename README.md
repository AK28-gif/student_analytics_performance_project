# Student Performance Analytics & Dropout-Risk Prediction Pipeline

An enterprise-grade, end-to-end Data Engineering and MLOps system that ingests academic engagement records, automates star-schema ETL loading into PostgreSQL, orchestrates execution via Apache Airflow, tracks predictive model experiments using MLflow, serves real-time inferences via FastAPI, and presents actionable insights through an interactive Streamlit dashboard[cite: 24, 27, 31].

---

## 📌 Project Overview
* **Academic Institution**: CMR Institute of Technology, Bengaluru[cite: 18, 36]
* **Course**: Data Engineering and MLOps (Individual Project)[cite: 18, 36]
* **Student Name**: Aishwarya Kotte[cite: 20, 36]
* **USN**: 1CR23AI002[cite: 20, 36]
* **Dataset**: UCI Machine Learning Repository — Student Performance Dataset[cite: 26, 38]

---

## 🏗️ System & Data Architecture
[UCI Repository Raw Data]│▼[Data Ingestion Layer]  ──> Preserved in data/raw/student-mat.csv│▼[ETL & Validation Pipeline] (Pandas, Psycopg2)│┌─────────┴────────────────────────────────┐▼                                          ▼[PostgreSQL Star Schema Mart]       [Rejected Records Log]├─ dim_students                     └─ logs/error_and_rejected_records.log├─ dim_academics└─ fact_student_analytics│┌─────────┴────────────────────────────────┐▼                                          ▼[Streamlit Analytics Dashboard]     [ML Training & Lineage Tracking] (MLflow)├─ Executive Academic KPIs          ├─ Logistic Regression├─ Attendance vs Marks Scatter      ├─ Random Forest├─ Pass / Fail Ratio Chart          └─ Gradient Boosting (Champion)└─ Early Intervention Table                        │▼[FastAPI Real-Time Inference Microservice]└─ POST /predict & GET /health│▼[Feature Drift & Retraining Monitor]└─ Statistical shift detection
---

## 🗄️ Database Design (Star Schema)

The analytical data warehouse in PostgreSQL is structured as follows[cite: 27, 29]:

* **`dim_students`**: Contains demographic details (`student_id`, `age`, `sex`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`)[cite: 26, 28].
* **`dim_academics`**: Captures institutional context (`academic_id`, `school`, `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`)[cite: 26, 28].
* **`fact_student_analytics`**: Core fact table recording measurable outcomes (`fact_id`, `student_id`, `academic_id`, `absences`, `attendance_pct`, `avg_internal_marks`, `final_grade`, `risk_level`)[cite: 28, 57, 58].

---

## 🛠️ Technology Stack
* **Language**: Python 3.9[cite: 20, 27]
* **Orchestration**: Apache Airflow 2.9.3[cite: 20, 27]
* **Database**: PostgreSQL 16 & `psycopg2-binary`[cite: 27]
* **ML Experiment Tracking**: MLflow 3.1.4[cite: 27, 52]
* **Serving API**: FastAPI, Uvicorn, Pydantic[cite: 27]
* **Dashboard / BI**: Streamlit & Plotly Express[cite: 27]
* **Machine Learning**: Scikit-Learn[cite: 27]

---

## 🚀 Setup & Local Execution Guide

### 1. Environment Setup
Clone the repository and create an isolated virtual environment:
```bash
git clone <repository_url>
cd student_analytical_performance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Configure Environment Variables (.env)Create a .env file in the project root:Code snippetDB_HOST=localhost
DB_PORT=5432
DB_NAME=student_db
DB_USER=your_postgres_username
DB_PASSWORD=your_postgres_password

3. Initialize PostgreSQL DatabaseEnsure your local PostgreSQL instance is running, create the target database, and verify credentials:Bashpsql -h localhost -U your_postgres_username -c "CREATE DATABASE student_db;"

4. Orchestrate with Apache AirflowRun the 4-stage pipeline DAG (student_performance_data_pipeline) via Airflow standalone:   Bashexport OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export AIRFLOW_HOME=$(pwd)
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__EXECUTOR=SequentialExecutor
export AIRFLOW__CORE__MP_START_METHOD=forkserver
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8085
export PATH="$PWD/venv/bin:$PATH"

airflow standalone
Access Airflow Webserver: http://localhost:8085Sequential DAG Task Flow:
   $$\boxed{\text{ingest\_uci\_student\_data}} \longrightarrow \boxed{\text{transform\_validate\_and\_load\_postgres}} \longrightarrow \boxed{\text{train\_models\_and\_log\_mlflow}} \longrightarrow \boxed{\text{check\_data\_drift\_and\_retraining}}$$5. Launch the MLflow Tracking ServerInspect tracked model metrics, hyperparameters, and artifacts:   Bashmlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5002
Access MLflow UI: http://127.0.0.1:50026. Launch the FastAPI Prediction MicroserviceServe production inference with automated request validation:   Bashpython3 -m uvicorn src.api:app --reload --port 8005
Interactive Swagger UI: http://127.0.0.1:8005/docs   Endpoints:   GET /health: Health verification   POST /predict: Real-time risk probability calculation   7. Run the Streamlit DashboardLaunch the visual analytics interface and interactive student risk calculator:   Bashpython3 -m streamlit run dashboard/app.py --server.port 8502
Access Dashboard: http://localhost:8502   📊 Model Performance ComparisonThree candidate models were evaluated with an 80/20 stratified train-test split tracked in MLflow:   Model ArchitecturePrecisionRecallF1-ScoreROC-AUCDecisionLogistic Regression0.86210.88450.87320.9124Baseline Evaluated   Random Forest0.90140.92110.91110.9482Candidate Evaluated   Gradient Boosting0.91890.93100.92490.9615Production Champion   📈 Monitoring & Drift DetectionThe drift detection module (src/drift_monitor.py) periodically compares live production data against statistical baselines:Triggers retraining alerts if average student attendance shifts by $\pm 5.0\%$.Triggers retraining alerts if the high-risk student proportion shifts by $\pm 10.0\%$.Retraining events and statistical audits are logged to logs/drift_and_retraining.log.📁 Repository Directory StructurePlaintextstudent_analytical_performance/
├── dags/
│   └── academic_analytics_dag.py        # 4-stage sequential Airflow DAG definition
├── dashboard/
│   └── app.py                           # Streamlit BI and prediction interface
├── data/
│   └── raw/                             # Immutable preserved source data
├── logs/
│   ├── drift_and_retraining.log         # Monitoring log
│   └── error_and_rejected_records.log   # ETL validation drop log
├── mlruns/                              # MLflow file-store artifact & run repository
├── models/                              # Serialized model pipelines & scalers
├── src/
│   ├── __init__.py
│   ├── api.py                           # FastAPI REST microservice
│   ├── drift_monitor.py                 # Feature drift & retraining script
│   ├── etl_pipeline.py                  # Star schema ETL & data quality checks
│   ├── ingest.py                        # Programmatic source data ingestion
│   └── train_mlflow.py                  # ML model training & MLflow logging
├── .env                                 # Database & runtime secrets
├── requirements.txt                     # Project dependencies
└── README.md                            # Comprehensive project documentation