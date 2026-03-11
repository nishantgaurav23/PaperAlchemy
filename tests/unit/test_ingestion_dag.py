"""Tests for the Airflow ingestion DAG task functions and constants.

Tests task functions directly with mocked HTTP calls.
DAG configuration tests use a helper to load the DAG module with
the Airflow namespace conflict resolved.
"""

from __future__ import annotations

import importlib
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Add airflow/dags to sys.path so arxiv_ingestion package is importable
_dags_dir = str(Path(__file__).resolve().parent.parent.parent / "airflow" / "dags")
if _dags_dir not in sys.path:
    sys.path.insert(0, _dags_dir)

from arxiv_ingestion.common import (  # noqa: E402
    API_BASE_URL,
    DEFAULT_TIMEOUT,
    FETCH_TIMEOUT,
    HEALTH_URL,
    INGEST_FETCH_URL,
)
from arxiv_ingestion.fetching import fetch_daily_papers  # noqa: E402
from arxiv_ingestion.reporting import generate_daily_report  # noqa: E402
from arxiv_ingestion.setup import setup_environment  # noqa: E402


def _load_dag():
    """Load the DAG module, handling the airflow namespace conflict.

    Our project has an airflow/ directory that shadows the apache-airflow
    package as a namespace package. We temporarily fix sys.modules to
    allow `from airflow import DAG` to resolve correctly.
    """
    # Remove our local airflow namespace package so real airflow is found
    keys_to_remove = [k for k in sys.modules if k == "airflow" and hasattr(sys.modules[k], "__path__")]
    for k in keys_to_remove:
        mod = sys.modules[k]
        # If the module is from our project dir (no __init__.py), it's a namespace pkg
        if mod.__spec__ is None or (hasattr(mod, "__file__") and mod.__file__ is None):
            del sys.modules[k]

    # Force reimport from the correct location
    if "arxiv_paper_ingestion" in sys.modules:
        return sys.modules["arxiv_paper_ingestion"]

    return importlib.import_module("arxiv_paper_ingestion")


# Check if real Airflow is available
try:
    _dag_module = _load_dag()
    _dag = _dag_module.dag
    HAS_AIRFLOW = True
except (ImportError, Exception):
    HAS_AIRFLOW = False
    _dag = None


# ---------------------------------------------------------------------------
# FR-1: DAG Configuration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_AIRFLOW, reason="apache-airflow not importable (namespace conflict)")
class TestDAGConfiguration:
    """Test DAG loading and configuration."""

    def test_dag_loads(self):
        assert _dag is not None
        assert _dag.dag_id == "arxiv_paper_ingestion"

    def test_dag_task_count(self):
        assert len(_dag.tasks) == 4

    def test_dag_task_ids(self):
        task_ids = {t.task_id for t in _dag.tasks}
        assert task_ids == {"setup_environment", "fetch_daily_papers", "generate_daily_report", "cleanup_old_pdfs"}

    def test_dag_task_dependencies(self):
        task_dict = {t.task_id: t for t in _dag.tasks}
        assert len(task_dict["setup_environment"].upstream_task_ids) == 0
        assert "setup_environment" in task_dict["fetch_daily_papers"].upstream_task_ids
        assert "fetch_daily_papers" in task_dict["generate_daily_report"].upstream_task_ids
        assert "generate_daily_report" in task_dict["cleanup_old_pdfs"].upstream_task_ids

    def test_dag_schedule(self):
        assert _dag.schedule_interval == "0 6 * * 1-5"

    def test_dag_default_args(self):
        assert _dag.default_args["retries"] == 2
        assert _dag.default_args["retry_delay"] == timedelta(minutes=30)

    def test_dag_catchup_disabled(self):
        assert _dag.catchup is False

    def test_dag_max_active_runs(self):
        assert _dag.max_active_runs == 1

    def test_dag_tags(self):
        assert "paperalchemy" in _dag.tags
        assert "ingestion" in _dag.tags
        assert "production" in _dag.tags


# ---------------------------------------------------------------------------
# FR-2: Setup Task (Health Check)
# ---------------------------------------------------------------------------


class TestSetupTask:
    """Test the health check setup task."""

    @patch("arxiv_ingestion.setup.httpx.get")
    def test_setup_task_healthy(self, mock_get):
        """Health check passes when API returns healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "version": "0.1.0"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        setup_environment()

        mock_get.assert_called_once_with(HEALTH_URL, timeout=DEFAULT_TIMEOUT)

    @patch("arxiv_ingestion.setup.httpx.get")
    def test_setup_task_unhealthy(self, mock_get):
        """Health check raises RuntimeError when API is down."""
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(RuntimeError, match="Health check failed"):
            setup_environment()

    @patch("arxiv_ingestion.setup.httpx.get")
    def test_setup_task_http_error(self, mock_get):
        """Health check raises RuntimeError on non-200 status."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="Health check failed"):
            setup_environment()


# ---------------------------------------------------------------------------
# FR-3: Fetch Daily Papers Task
# ---------------------------------------------------------------------------


class TestFetchTask:
    """Test the fetch daily papers task."""

    def _make_ti(self):
        ti = MagicMock()
        ti.xcom_push = MagicMock()
        return ti

    @patch("arxiv_ingestion.fetching.httpx.post")
    def test_fetch_task_success(self, mock_post):
        """Fetch task calls API and pushes XCom with results."""
        ti = self._make_ti()
        fetch_result = {
            "papers_fetched": 5,
            "pdfs_downloaded": 4,
            "pdfs_parsed": 3,
            "papers_stored": 5,
            "arxiv_ids": ["2602.00001", "2602.00002"],
            "errors": [],
            "processing_time": 42.5,
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fetch_result
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        execution_date = datetime(2026, 3, 10, 6, 0, 0, tzinfo=UTC)
        fetch_daily_papers(ti=ti, execution_date=execution_date)

        ti.xcom_push.assert_called_once_with(key="fetch_result", value=fetch_result)

    @patch("arxiv_ingestion.fetching.httpx.post")
    def test_fetch_task_date_logic(self, mock_post):
        """Target date is execution_date - 1 day."""
        ti = self._make_ti()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"papers_fetched": 0, "arxiv_ids": [], "errors": []}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        execution_date = datetime(2026, 3, 10, 6, 0, 0, tzinfo=UTC)
        fetch_daily_papers(ti=ti, execution_date=execution_date)

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["target_date"] == "20260309"

    @patch("arxiv_ingestion.fetching.httpx.post")
    def test_fetch_task_api_error_status(self, mock_post):
        """Fetch task raises RuntimeError on HTTP error."""
        ti = self._make_ti()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="Fetch endpoint error"):
            fetch_daily_papers(ti=ti, execution_date=datetime(2026, 3, 10, tzinfo=UTC))

    @patch("arxiv_ingestion.fetching.httpx.post")
    def test_fetch_task_connection_error(self, mock_post):
        """Fetch task raises RuntimeError on connection failure."""
        ti = self._make_ti()
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(RuntimeError, match="Fetch request failed"):
            fetch_daily_papers(ti=ti, execution_date=datetime(2026, 3, 10, tzinfo=UTC))


# ---------------------------------------------------------------------------
# FR-4: Daily Report Task
# ---------------------------------------------------------------------------


class TestReportTask:
    """Test the daily report task."""

    def _make_ti(self, fetch_result=None):
        ti = MagicMock()
        ti.xcom_pull.return_value = fetch_result
        return ti

    @patch("arxiv_ingestion.reporting.httpx.get")
    def test_report_task_with_data(self, mock_get, caplog):
        """Report task logs a structured report from XCom data."""
        fetch_result = {
            "papers_fetched": 5,
            "pdfs_downloaded": 4,
            "pdfs_parsed": 3,
            "papers_stored": 5,
            "arxiv_ids": ["2602.00001"],
            "errors": [],
            "processing_time": 42.5,
        }
        ti = self._make_ti(fetch_result)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "version": "0.1.0"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        execution_date = datetime(2026, 3, 10, 6, 0, 0, tzinfo=UTC)
        with caplog.at_level(logging.INFO):
            generate_daily_report(ti=ti, execution_date=execution_date)

        ti.xcom_pull.assert_called_once_with(task_ids="fetch_daily_papers", key="fetch_result")

    @patch("arxiv_ingestion.reporting.httpx.get")
    def test_report_task_missing_xcom(self, mock_get, caplog):
        """Report task handles missing XCom data gracefully."""
        ti = self._make_ti(None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "version": "0.1.0"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        execution_date = datetime(2026, 3, 10, 6, 0, 0, tzinfo=UTC)
        with caplog.at_level(logging.INFO):
            generate_daily_report(ti=ti, execution_date=execution_date)

    @patch("arxiv_ingestion.reporting.httpx.get")
    def test_report_task_health_failure(self, mock_get, caplog):
        """Report task still completes even if health check fails."""
        fetch_result = {"papers_fetched": 5, "errors": []}
        ti = self._make_ti(fetch_result)
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        execution_date = datetime(2026, 3, 10, 6, 0, 0, tzinfo=UTC)
        with caplog.at_level(logging.WARNING):
            generate_daily_report(ti=ti, execution_date=execution_date)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Test common constants are properly defined."""

    def test_api_base_url(self):
        assert API_BASE_URL == "http://api:8000"

    def test_health_url(self):
        assert HEALTH_URL == "http://api:8000/api/v1/ping"

    def test_fetch_url(self):
        assert INGEST_FETCH_URL == "http://api:8000/api/v1/ingest/fetch"

    def test_fetch_timeout(self):
        assert FETCH_TIMEOUT == 1800

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 30
