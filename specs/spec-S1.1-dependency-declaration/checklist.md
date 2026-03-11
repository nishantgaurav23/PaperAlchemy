# Checklist S1.1 — Dependency Declaration

## Phase 1: Setup
- [x] Read current pyproject.toml
- [x] Identify deps to add, update, and remove

## Phase 2: Implementation
- [x] Update project metadata (name, version, description, license)
- [x] Add production dependencies (web, db, search, llm, embeddings, cache, eval)
- [x] Replace sync drivers with async (psycopg2-binary → asyncpg, requests → httpx)
- [x] Add langchain-google-genai for Gemini 3 Flash
- [x] Add ragas for evaluation framework
- [x] Remove unused deps (gradio, telegram-bot, types-sqlalchemy)
- [x] Update dev dependencies
- [x] Configure ruff lint rules (select, format)
- [x] Configure mypy with pydantic plugin
- [x] Configure pytest settings

## Phase 3: Verification
- [x] `uv sync` succeeds (exit code 0)
- [x] Key imports work: fastapi, sqlalchemy, langchain, langgraph, ragas, langchain_google_genai
- [x] `ruff check` runs without config errors
- [x] `pytest --co` runs without config errors
- [x] No version conflicts in dependency resolution

## Phase 4: Notebook
- [x] Create `notebooks/specs/S1.1_dependency.ipynb` verifying imports and tool config

## Phase 5: Roadmap Update
- [x] Update roadmap.md S1.1 status to `done`
