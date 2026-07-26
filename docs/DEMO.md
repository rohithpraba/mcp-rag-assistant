# Public web demo

The browser and read-only REST API run in one FastAPI container. They call the
existing retrieval and answering services against the fixed `public-demo`
Chroma workspace. Chroma and the Hugging Face cache use Docker volumes. Ollama
remains on the host and is reached at `host.docker.internal:11434`; it is never
proxied or published.

## One-click VS Code

Open the repository folder in VS Code and press `Ctrl+Shift+B`. The default
build task starts the complete public demo. It creates and validates a fresh
Quick Tunnel URL, copies the URL to the Windows clipboard, and opens it in the
default browser. The temporary hostname changes after tunnel recreation.

## Double-click Windows

- `run-demo.cmd` starts the public demo.
- `stop-demo.cmd` stops web and cloudflared while preserving volumes.
- `demo-status.cmd` shows Docker, Ollama, model, local, and tunnel status.

## Command line

Local:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo.ps1 start
```

Public:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo.ps1 start -Public
```

Stop:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo.ps1 stop
```

Status:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo.ps1 status
```

Prerequisites:

- Docker Desktop installed
- Ollama installed
- `gemma3:latest` available locally
- internet access for a Quick Tunnel
- outbound TCP port 7844 available for HTTP/2 tunnel transport

The equivalent direct Compose commands are:

```bash
docker compose up --build web
docker compose --profile public up --build
```

The local UI is at `http://127.0.0.1:8000`. Quick Tunnels exist only while the
laptop, Docker services, Ollama, and tunnel are running and have no uptime
guarantee.

Anonymous access is read-only. Routes are `GET /`, `/healthz`, `/readyz`,
`/api/v1/demo/sources` and `POST /api/v1/search`, `/api/v1/ask`. There are no
upload, ingestion, refresh, deletion, URL-fetch, or Ollama-proxy routes.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What technologies does this project use?","top_k":3}'
```

Inputs are bounded, requests are rate limited, and only one generation runs at
a time by default. Generation has a strict response timeout; because the RAG
call is synchronous in a worker thread, a timed-out thread cannot be forcibly
stopped once Ollama work has begun.

On Windows Docker Desktop, enable host networking integration if
`host.docker.internal` does not resolve and verify Ollama accepts connections
from Docker. On Linux, Compose supplies the `host-gateway` mapping. Do not
publish port 11434.

This is not a 24/7 service and the project does not claim production
readiness.
