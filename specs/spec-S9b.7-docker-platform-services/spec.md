# S9b.7 — Docker Platform Services

## Summary
Add Docker Compose services required by P22 (Paper-to-Code) and P23 (Audio/Podcast) features:
- **MinIO** (S3-compatible object storage) for audio files, code artifacts, and general file storage
- **Code sandbox** (sysbox-based secure container) for P22 sandboxed code execution
- Update **API service** environment with new platform env vars (Anthropic, TTS, Auth)

## Depends On
- **S1.3** (Docker Infrastructure) — base compose.yml with core services ✅

## Downstream Dependents
- **S11b.1** (Platform Monitoring) — health checks for MinIO, sandbox
- **S11b.2** (Platform Ops Guide) — MinIO admin, sandbox lifecycle docs
- **S11b.3** (Platform Deploy) — Cloud Storage replaces MinIO in prod
- **S22.3** (Code Sandbox) — uses sandbox service
- **S23.2** (Text-to-Speech) — stores audio in MinIO

## Files Modified
| File | Change |
|------|--------|
| `compose.yml` | Add MinIO, sandbox services; update API env vars; add volumes |
| `.env.example` | Add MINIO__* env vars (endpoint, access key, secret key, bucket) |
| `Makefile` | Add `platform` profile target |

## Requirements

### R1: MinIO Service
- Image: `minio/minio` (latest)
- Profile: `platform` (not started by default)
- Ports: `9100:9000` (API), `9101:9001` (Console) — avoid conflict with Langfuse MinIO
- Default bucket: `paperalchemy` (auto-created via entrypoint)
- Credentials: `MINIO_ROOT_USER=paperalchemy`, `MINIO_ROOT_PASSWORD=paperalchemy_minio_secret`
- Volume: `minio_data:/data`
- Health check: `mc ready local`
- Network: `paperalchemy-network`

### R2: Code Sandbox Service
- Image: `docker:27-dind` (Docker-in-Docker for sandboxed execution)
- Profile: `platform` (not started by default)
- Privileged: true (required for DinD)
- Volume: `sandbox_data:/var/lib/docker` (isolated Docker storage)
- Environment: `DOCKER_TLS_CERTDIR=` (disable TLS for local dev)
- Exposed port: `2376` internally (API service connects via `sandbox:2376`)
- Health check: `docker info`
- Network: `paperalchemy-network`

### R3: API Service Environment Updates
Add to API service `environment` block:
- `MINIO__ENDPOINT=minio:9000`
- `MINIO__ACCESS_KEY=paperalchemy`
- `MINIO__SECRET_KEY=paperalchemy_minio_secret`
- `MINIO__BUCKET=paperalchemy`
- `MINIO__USE_SSL=false`
- `SANDBOX__HOST=sandbox`
- `SANDBOX__PORT=2376`
- Forward from .env: `ANTHROPIC__API_KEY`, `AUTH__SECRET_KEY`, `TTS__PROVIDER`, `TTS__VOICE_MAP`

### R4: .env.example Updates
Add MinIO and sandbox env vars:
```
# MinIO (S3-compatible object storage)
MINIO__ENDPOINT=localhost:9100
MINIO__ACCESS_KEY=paperalchemy
MINIO__SECRET_KEY=paperalchemy_minio_secret
MINIO__BUCKET=paperalchemy
MINIO__USE_SSL=false

# Code Sandbox
SANDBOX__HOST=localhost
SANDBOX__PORT=2376
```

### R5: Makefile Platform Target
Add `platform` target to bring up platform services:
```makefile
platform:  ## Start platform services (MinIO, sandbox)
	docker compose --profile platform up -d
```

## TDD Plan

### Tests (Red → Green)
1. **test_compose_platform_services** — Parse compose.yml, verify `minio` and `sandbox` services exist
2. **test_minio_service_config** — Verify MinIO image, ports, profile, healthcheck, volume, network
3. **test_sandbox_service_config** — Verify DinD image, profile, privileged, healthcheck, volume, network
4. **test_api_env_vars** — Verify API service has MinIO, sandbox, Anthropic, TTS, Auth env vars
5. **test_env_example_minio_vars** — Verify .env.example contains MINIO__* vars
6. **test_env_example_sandbox_vars** — Verify .env.example contains SANDBOX__* vars
7. **test_makefile_platform_target** — Verify Makefile has `platform` target
8. **test_volumes_declared** — Verify `minio_data` and `sandbox_data` volumes in compose.yml

## Tangible Outcomes
- `docker compose --profile platform config` validates without errors
- MinIO console accessible at `http://localhost:9101` when platform profile active
- `paperalchemy` bucket auto-created on MinIO startup
- DinD sandbox service healthy and accepting Docker API calls
- API service can resolve MinIO and sandbox hostnames on `paperalchemy-network`
