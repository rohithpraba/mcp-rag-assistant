# Changelog

All notable repository-maintenance changes are documented here.

## Unreleased

### Added

- Standard Python packaging through `pyproject.toml`.
- Installed command-line entry points for indexing, retrieval, grounded answering, and the MCP server.
- Repository license, third-party notices, security policy, and contribution guidance.

### Changed

- Dependency groups are organized around core, web, development, and optional training use cases.
- Continuous integration and container publishing are being aligned with installable-package validation and immutable commit tags.

### Preserved

- Existing RAG, MCP, web-demo, and LoRA evaluation behaviour.
- Exact `INSUFFICIENT_EVIDENCE` handling.
- Reported evaluation sizes and limitations.
