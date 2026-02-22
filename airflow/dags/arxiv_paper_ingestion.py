"""
The DAG definition. This is the only file AirFlow actually reads directly. It wires all 5 tasks togetehr with their
scheudle, retry config, and dependencies.

PaperAlchemy — Daily arXiv Paper Ingestion DAG

Schedule: Monday–Friday at 6 AM UTC
Pipeline: setup → fetch → index → report → cleanup

Each run processes papers published on the previous calendar day.
Max active runs = 1: prevents two runs from racing on the same date.
"""
import sys
from pathlib import Path

# Make src/ importable inside the Airflow container
sys.path.insert(0, "/opt/airflow/project")

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from arxiv_ingestion.setup import setup_environment
from arxiv_ingestion.fetching import fetch_daily_papers
from arxiv_ingestion.indexing import index_papers_hybrid
from arxiv_ingestion.reporting import generate_daily_report

# ---------------------------------------------------------------------------
# Default arguments applied to every task
# ---------------------------------------------------------------------------
default_args = {
    "owner": "paperalchemy",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
    "start_date": datetime(2026, 1, 1)
}

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="arxiv_paper_ingestion",
    default_args=default_args,
    description="Fetch, parse, and index daily arXiv CS.AI papers",
    schedule_interval="0 6 * * 1-5",  # Mon–Fri at 06:00 UTC
    catchup=False,                    # Don;t backfill missed runs
    max_active_runs=1,                # One run at a time
    tags=["paperalchemy", "ingestion", "production"],
)as dag:
    
    # Task 1 - Verifiy services are healthy
    t1_setup = PythonOperator(
        task_id="setup_environment",
        python_callable=setup_environment,
    )

    # Task 2 - Fetch yesterday's papers from arXiv -> PostgreSQL
    t2_fetch = PythonOperator(
        task_id="fetch_daily_papers",
        python_callable=fetch_daily_papers,
    )

    # Task 3 - Chunk + embed + index -> OpenSearch
    t3_index = PythonOperator(
        task_id="index_papers_hybrid",
        python_callable=index_papers_hybrid,
    )

    # Task 4 - Collect stats and log report
    t4_report = PythonOperator(
        task_id="generate_daily_report",
        python_callable=generate_daily_report,
    )

    # Task 5 - Remove cached PDFs older than 30 days
    t5_cleanup = BashOperator(
        task_id="cleanup_temp_files",
        bash_command=(
            "find /tmp/paperalchemy_pdfs -type f -name '*.pdf' "
            "-mtime +30 -delete 2>/dev/null || true"
        ),
    )

    # ---------------------------------------------------------------------------
    # Task dependencies — strict linear pipeline
    # ---------------------------------------------------------------------------
    t1_setup >> t2_fetch >> t3_index >> t4_report >> t5_cleanup


