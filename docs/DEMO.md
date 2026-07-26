# Public web demo

The browser and read-only REST API run in one FastAPI container. They call the
existing retrieval and answering services against the fixed `public-demo`
Chroma workspace. Chroma and the Hugging Face cache use Docker volumes. Ollama
remains on the host and is reached at `host.docker.internal:11434`; it is never
proxied or published.

Start Ollama on the host and ensure `gemma3:latest` is installed, then run:

```bash
docker compose up --build web
docker compose --profile public up --build
```

The local UI is at `http://localhost:8000`. For public mode, find the temporary
`https://….trycloudflare.com` address in the `cloudflared` logs. Quick Tunnels
exist only while the laptop, Docker services, Ollama, and tunnel are running
and have no uptime guarantee.

Anonymous access is read-only. Routes are `GET /`, `/healthz`, `/readyz`,
`/api/v1/demo/sources` and `POST /api/v1/search`, `/api/v1/ask`. There are no
upload, ingestion, refresh, deletion, URL-fetch, or Ollama-proxy routes.

```bash
curl -X POST http://localhost:8000/api/v1/ask \
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
readiness. Oracle Always Free is only a possible future optional 24/7
experiment; it is not part of this demo.
