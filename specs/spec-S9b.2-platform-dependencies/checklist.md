# Checklist S9b.2 — Platform Dependency Declaration

## Phase 1: Setup
- [x] Read current pyproject.toml and .env.example
- [x] Identify new deps to add (anthropic, jose, passlib, websockets, edge-tts, python-pptx)

## Phase 2: Tests (Red)
- [x] Write `tests/unit/test_platform_deps.py` with import tests for all new packages
- [x] Verify tests fail (packages not yet installed)

## Phase 3: Implementation (Green)
- [x] Add Anthropic SDK to pyproject.toml
- [x] Add auth libraries (python-jose[cryptography], passlib[bcrypt]) to pyproject.toml
- [x] Add websockets to pyproject.toml
- [x] Add edge-tts to pyproject.toml
- [x] Add python-pptx to pyproject.toml
- [x] Run `uv sync` — verify exit code 0
- [x] Update .env.example with Anthropic config section
- [x] Update .env.example with Auth config section
- [x] Update .env.example with TTS config section
- [x] Verify all import tests pass

## Phase 4: Verification
- [x] `uv sync` succeeds
- [x] All new imports work (9/9 tests pass)
- [x] Existing tests still pass (993 passed, 4 pre-existing failures in arxiv date tests)
- [x] `ruff check` passes
- [x] No version conflicts

## Phase 5: Notebook
- [x] Create `notebooks/specs/S9b.2_platform_deps.ipynb`

## Phase 6: Roadmap Update
- [x] Update roadmap.md S9b.2 status to `done`
- [x] Append spec summary to docs/spec-summaries.md
