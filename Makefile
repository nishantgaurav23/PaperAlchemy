.PHONY: help up up-all up-langfuse down down-clean build start stop restart status logs health setup format lint test test-cov gradio clean

# Default target
help:
	@echo "PaperAlchemy - Available Commands"
	@echo "================================="
	@echo ""
	@echo "Docker:"
	@echo "  make up            - Start core services (postgres, redis, opensearch)"
	@echo "  make up-all        - Start ALL services (core + full + langfuse + dev-tools)"
	@echo "  make up-langfuse   - Start core + Langfuse observability stack"
	@echo "  make down          - Stop all services"
	@echo "  make down-clean    - Stop all services and remove volumes"
	@echo "  make build         - Build API Docker image"
	@echo "  make start         - Start all services (legacy, same as up-all)"
	@echo "  make stop          - Stop all services (legacy, same as down)"
	@echo "  make restart       - Restart all services"
	@echo "  make status        - Show service status"
	@echo "  make logs          - Show service logs (use s=<service> for specific)"
	@echo "  make health        - Check all services health"
	@echo ""
	@echo "Development:"
	@echo "  make setup         - Install Python dependencies"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint and type check"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make gradio        - Start Gradio web UI (http://localhost:7861)"
	@echo "  make clean         - Clean up everything"

# ── Docker commands ──────────────────────────────────────────────────────────

# Core services only (postgres, redis, opensearch)
up:
	docker compose up -d

# All services: core + full + langfuse + dev-tools
up-all:
	docker compose --profile full --profile langfuse --profile dev-tools up -d

# Core + Langfuse observability stack
up-langfuse:
	docker compose --profile langfuse up -d

# Build API image
build:
	docker compose build api

# Stop services
down:
	docker compose --profile full --profile langfuse --profile dev-tools down

# Stop + remove volumes
down-clean:
	docker compose --profile full --profile langfuse --profile dev-tools down -v

# Legacy aliases
start:
	docker compose --profile full --profile langfuse --profile dev-tools up --build -d

stop:
	docker compose --profile full --profile langfuse --profile dev-tools down

restart:
	docker compose --profile full --profile langfuse --profile dev-tools down && docker compose --profile full --profile langfuse --profile dev-tools up --build -d

status:
	docker compose ps -a

# Logs (use: make logs or make logs s=api)
logs:
ifdef s
	docker compose logs -f $(s)
else
	docker compose logs -f
endif

# Health check
health:
	@echo "Checking PostgreSQL..."
	@docker compose exec -T postgres pg_isready -U paperalchemy || echo "PostgreSQL not ready"
	@echo "\nChecking OpenSearch..."
	@curl -sf http://localhost:9201/_cluster/health | head -5 || echo "OpenSearch not running"
	@echo "\nChecking Redis..."
	@docker compose exec -T redis redis-cli ping || echo "Redis not running"
	@echo "\nChecking API..."
	@curl -sf http://localhost:8000/health || echo "API not running"

# ── Development ──────────────────────────────────────────────────────────────

setup:
	uv sync

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests
	uv run mypy src

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=html

# Gradio UI
gradio:
	uv run python gradio_launcher.py

# Cleanup
clean:
	docker compose --profile full --profile langfuse --profile dev-tools down -v
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
