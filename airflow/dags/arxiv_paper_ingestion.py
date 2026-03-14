"""Airflow DAG: arXiv paper ingestion pipeline.

Pipeline: setup → fetch → report

Schedule: Mon-Fri at 06:00 UTC
Retries: 2, with 30-minute delay
Catchup: disabled
Max active runs: 1

PDFs are cached permanently in data/arxiv_pdfs/ (managed by the API).
No cleanup task — PDFs are cheap to store and expensive to re-download.

All task functions call the PaperAlchemy REST API via HTTP to avoid
SQLAlchemy version conflicts between Airflow and the application.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the dags directory to sys.path so arxiv_ingestion is importable.
# In Airflow, DAGs are loaded from the dags folder, not as a Python package.
_dags_dir = str(Path(__file__).resolve().parent)
if _dags_dir not in sys.path:
    sys.path.insert(0, _dags_dir)

from airflow import DAG  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402
from arxiv_ingestion.fetching import fetch_daily_papers  # noqa: E402
from arxiv_ingestion.reporting import generate_daily_report  # noqa: E402
from arxiv_ingestion.setup import setup_environment  # noqa: E402

default_args = {
    "owner": "paperalchemy",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
    "start_date": datetime(2026, 1, 1),
}

with DAG(
    dag_id="arxiv_paper_ingestion",
    default_args=default_args,
    description="Fetch, parse, and index daily arXiv CS.AI papers",
    schedule_interval="0 6 * * 1-5",
    catchup=False,
    max_active_runs=1,
    tags=["paperalchemy", "ingestion", "production"],
) as dag:
    t1_setup = PythonOperator(
        task_id="setup_environment",
        python_callable=setup_environment,
    )

    t2_fetch = PythonOperator(
        task_id="fetch_daily_papers",
        python_callable=fetch_daily_papers,
        provide_context=True,
    )

    t3_report = PythonOperator(
        task_id="generate_daily_report",
        python_callable=generate_daily_report,
        provide_context=True,
    )

    t1_setup >> t2_fetch >> t3_report
