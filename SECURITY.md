# Security policy

## Project boundary

This repository is a local-first portfolio/research implementation. It is not an authenticated multi-user service and is not presented as production-ready.

The public demo is intentionally read-only and uses a fixed workspace. Do not expose arbitrary file indexing, URL ingestion, workspace administration, or model-management operations through a public endpoint without a separate security review.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to `rohithpraba03@gmail.com` before opening a public issue. Include:

- the affected component and version or commit;
- reproduction steps;
- expected and observed behaviour;
- impact and suggested mitigation, where known.

Do not include credentials, private documents, model weights, databases, vector stores, or personal data in a report.

## Supported version

Security fixes are evaluated against the current `main` branch. Historical commits and temporary demonstration URLs are not maintained as supported releases.

## Known operational limits

- The MCP server uses local standard input/output and has no remote authentication layer.
- The FastAPI demonstration uses process-local rate limiting and is not designed for horizontally scaled deployment.
- Citation-label validation checks whether labels were supplied and recognized; it does not prove semantic entailment of every generated sentence.
- Public URL ingestion requires continued review before use in hostile or multi-tenant environments.
- Model and dependency security also depends on upstream packages, model artifacts, and local runtime configuration.

## Secret handling

Never commit `.env` files, tokens, credentials, private documents, local indexes, model artifacts, or generated databases. Rotate any secret immediately if it is accidentally published, then remove it through an approved history-remediation process.
