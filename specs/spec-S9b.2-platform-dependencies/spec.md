# Spec S9b.2 — Platform Dependency Declaration

## Overview
Add new Python dependencies required by phases P14–P23 (auth, real-time comments, code generation, audio/TTS, slide generation) and declare their corresponding environment variables. This is a pure configuration spec — no application code, only `pyproject.toml` and `.env.example` changes.

## Depends On
- S1.1 (Dependency Declaration) — `done`

## Location
- `pyproject.toml` (root)
- `.env.example` (root)

## Requirements

### R1: Anthropic SDK (for P22 Code Generation)
- Add `anthropic>=0.52.0` to production dependencies
- Purpose: Claude Opus/Sonnet/Haiku for paper-to-code generation agent (S22.2)
- Add `ANTHROPIC__API_KEY` and `ANTHROPIC__MODEL` to `.env.example`

### R2: Authentication Libraries (for P14 User Auth)
- Add `python-jose[cryptography]>=3.3.0` — JWT token encoding/decoding
- Add `passlib[bcrypt]>=1.7.4` — password hashing
- Purpose: User authentication and JWT sessions (S14.1)
- Add `AUTH__SECRET_KEY`, `AUTH__ALGORITHM`, `AUTH__ACCESS_TOKEN_EXPIRE_MINUTES` to `.env.example`

### R3: WebSocket Support (for P14 Real-time Comments)
- Add `websockets>=14.0` — WebSocket protocol support
- Purpose: Real-time inline comments on papers (S14.2)
- Note: FastAPI already supports WebSocket via Starlette, but explicit `websockets` package is needed for the underlying transport

### R4: Text-to-Speech (for P23 Audio/Podcast)
- Add `edge-tts>=7.0.0` — Free Microsoft Edge TTS (default provider, no API key needed)
- Purpose: Multi-voice podcast generation from paper overviews (S23.2)
- Add `TTS__PROVIDER` and `TTS__VOICE_MAP` to `.env.example`

### R5: Slide Generation (for P19 Advanced AI)
- Add `python-pptx>=1.0.0` — PowerPoint slide generation
- Purpose: AI-generated presentation slides from papers (S19.3)

### R6: Environment Variable Declarations
Update `.env.example` with all new configuration sections:

```env
# Anthropic Claude (cloud LLM for code generation)
ANTHROPIC__API_KEY=your_anthropic_api_key_here
ANTHROPIC__MODEL=claude-sonnet-4-20250514

# Authentication (JWT)
AUTH__SECRET_KEY=change-this-to-a-random-secret-key
AUTH__ALGORITHM=HS256
AUTH__ACCESS_TOKEN_EXPIRE_MINUTES=30

# Text-to-Speech
TTS__PROVIDER=edge-tts
TTS__VOICE_MAP={"speaker_1": "en-US-GuyNeural", "speaker_2": "en-US-JennyNeural"}
```

### R7: Dependency Resolution
- All new dependencies must resolve without conflicts alongside existing deps
- `uv sync` must succeed with exit code 0
- No version conflicts between new and existing packages

## Tangible Outcomes
1. `pyproject.toml` updated with 5 new dependency groups (anthropic, auth, websockets, tts, slides)
2. `.env.example` updated with new configuration sections
3. `uv sync` succeeds — all deps install cleanly
4. Key imports work: `import anthropic`, `from jose import jwt`, `from passlib.hash import bcrypt`, `import websockets`, `import edge_tts`, `from pptx import Presentation`
5. Existing tests still pass (no breaking changes)

## TDD Notes
- **Test file**: `tests/unit/test_platform_deps.py`
- **Red phase**: Write import tests that fail (packages not installed)
- **Green phase**: Add deps to `pyproject.toml`, run `uv sync`
- **Refactor phase**: Organize dependency groups with comments, verify `.env.example` completeness

## Notebook
- `notebooks/specs/S9b.2_platform_deps.ipynb` — verify all imports, print versions, validate env var parsing
