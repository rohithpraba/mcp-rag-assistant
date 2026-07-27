# Third-party notices

The MIT license in this repository applies to the original project code and documentation authored for this repository. It does not replace or modify the licenses, terms, or acceptable-use requirements of third-party software, models, datasets, or hosted services.

## Software dependencies

This project depends on open-source packages including Chroma, FastAPI, the Model Context Protocol Python SDK, PyMuPDF, Sentence Transformers, Hugging Face libraries, and their transitive dependencies. Each dependency remains governed by its own upstream license. The authoritative dependency list is declared in `pyproject.toml`.

## Models and model services

Model names such as Gemma and embedding-model identifiers refer to separately distributed model artifacts. No model weights are licensed by this repository. Users are responsible for reviewing the applicable model license and usage terms before downloading or deploying a model.

Ollama and optional Cloudflare tooling are external software or services. Their names are used only to describe interoperability; this repository does not redistribute or relicense them.

## Evaluation material

The committed evaluation collections are small, controlled project artifacts intended for regression and behaviour testing. Reported results must remain paired with their documented test-set sizes and limitations.

## Responsibility

Before redistributing a packaged application, container image, model artifact, or derived dataset, review the licenses and notices of every included component. Open an issue if an attribution or licensing detail appears incomplete.
