from ingest import run_ingestion
from etl_pipeline import run_etl

def main():
    print("=== STARTING ACADEMIC DATA PIPELINE ===")
    run_ingestion()
    run_etl()
    print("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()