# EU AI Act RAG Assistant

A research prototype for querying the European Union Artificial Intelligence Act (EU AI Act). It combines hybrid retrieval, cross-encoder reranking, local LLM generation, provision-level citations, and an NLI-based claim-verification stage.

> **Important:** This project is for research and educational purposes only. It is not legal advice, a legally validated compliance assessment, or a guarantee that every generated claim is correct. Verify important conclusions against the authoritative legislation.

## Features

- **Hybrid retrieval:** sparse BM25 search combined with dense vector search in Qdrant, fused with Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking:** `BAAI/bge-reranker-large` reranks retrieved passages before context construction.
- **Local LLM generation:** Ollama using `llama3.2:3b` for answer generation.
- **Citation-aware answering:** prompts, selected deterministic extraction paths, and citation normalization intended to keep claims tied to legal provisions.
- **NLI claim checks:** `cross-encoder/nli-deberta-v3-base` evaluates whether answer claims are supported by the retrieved evidence and flags some unsupported or uncited claims.
- **API and UI:** FastAPI backend and Streamlit frontend, including a Server-Sent Events endpoint for progressive status/token updates.
- **Docker Compose:** separate API and UI services for local containerized use.
- **Evaluation tools:** a benchmark question set and RAGAS-oriented evaluation script.

NLI results are an automated signal, not proof of legal correctness. The system can still retrieve irrelevant evidence, miss an exact provision, misattribute a citation, or incorrectly classify a claim.

## Architecture

```text
Browser
  |
  v
Streamlit UI (app.py)
  |  HTTP / Server-Sent Events
  v
FastAPI (src/api/main.py)
  |
  +--> BM25 sparse retrieval -----------------+
  |                                            |
  +--> BGE dense embeddings + local Qdrant ----+--> Reciprocal Rank Fusion
                                                   |
                                                   v
                                          Cross-encoder reranker
                                                   |
                                                   v
                                            Context builder
                                                   |
                                                   v
                                        Ollama LLM generation
                                                   |
                                                   v
                                   Citation handling / repair paths
                                                   |
                                                   v
                                        NLI claim verification
                                                   |
                                                   v
                               Final answer, sources, and audit to UI
```

## Technology Stack

- Python
- FastAPI, Uvicorn, Pydantic
- Streamlit and Requests
- Ollama (`llama3.2:3b`)
- Sentence Transformers / PyTorch
- Dense embedding model: `BAAI/bge-small-en-v1.5`
- Reranker: `BAAI/bge-reranker-large`
- NLI model: `cross-encoder/nli-deberta-v3-base`
- Qdrant local storage and BM25
- Docker and Docker Compose

The embedding, reranker, and NLI models are downloaded from Hugging Face when needed. First startup may therefore take longer and requires network access unless those models are already cached.

## Repository Layout

```text
eu_ai_act/
│
├── README.md
├── LICENSE
├── .gitignore
├── .dockerignore
├── requirements.txt
│
├── app.py
├── eval_rag.py
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
│
├── data/
│   ├── eval_dataset.py
│   │
│   ├── raw/
│   │   ├── eu_ai_act_2024_1689.html
│   │   ├── eu_ai_act_2024_1689_consolidated_20260727.html
│   │   └── gdpr_2016_679.html
│   │
│   ├── processed/
│   │   ├── bm25.pkl
│   │   └── chunks.json
│   │
│   └── qdrant_db/
│       ├── meta.json
│       └── collection/
│           └── eu_ai_act_clauses/
│               └── storage.sqlite
│
├── evaluation/
│   └── questions.json
│
├── scripts/
│   ├── build_index.py
│   ├── check_ollama_output.py
│   ├── eval_ragas_faithfulness.py
│   └── fetch_data.py
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── eval/
│   │   └── run_benchmark.py
│   │
│   ├── generation/
│   │   └── generation.py
│   │
│   ├── guardrails/
│   │   └── nli_verifier.py
│   │
│   ├── ingestion/
│   │   ├── parser.py
│   │   └── schemas.py
│   │
│   └── retrieval/
│       ├── context_builder.py
│       ├── hybrid_search.py
│       ├── legal_reference_router.py
│       └── reranker.py
│
└── tests/
    └── test_rag_upgrade.py
```

## Prerequisites

- Python 3.12 for the local commands below
- Docker Desktop with the Linux engine enabled, if using Docker Compose
- Ollama installed and running
- Enough RAM, CPU, and disk space for the embedding, reranker, NLI, and LLM models
- Network access on first run to download model weights, unless already cached

Pull the generation model if it is not already available:

```bash
ollama pull llama3.2:3b
```

The embedding, reranker, and NLI weights are managed by the Python libraries and may download when their components initialize.

## Run Locally (without Docker)

Run commands from the repository root.

### 1. Create and activate a virtual environment

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
```

Ensure Ollama is running and `llama3.2:3b` is available.

### 3. Start FastAPI

In the first terminal, from the project root:

```powershell
py -3.12 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

API docs: <http://localhost:8000/docs>  
Health check: <http://localhost:8000/health>

### 4. Start Streamlit

In a second terminal:

```powershell
py -3.12 -m streamlit run app.py
```

Open <http://localhost:8501>.

By default, the frontend calls the API at `http://localhost:8000`. The API base URL can be configured with the `API_BASE_URL` environment variable or through the UI, depending on the current frontend configuration.

## Run with Docker Compose

The provided Compose setup runs separate API and UI containers and mounts the local `data/` directory into the API container. The API connects to Ollama through `http://host.docker.internal:11434`, so Ollama must be running on the Docker host and the model must be available there.

```powershell
docker compose config
docker compose build
docker compose up -d
docker compose ps
```

Open:

- Streamlit UI: <http://localhost:8501>
- FastAPI docs: <http://localhost:8000/docs>
- FastAPI health: <http://localhost:8000/health>

The API initializes several models on startup. The first start can take time while weights are loaded or downloaded. Check logs with:

```powershell
docker compose logs -f api
docker compose logs -f ui
```

Stop the containers with:

```powershell
docker compose down
```

## Evaluation

### RAG benchmark

Run from the repository root:

```powershell
py -3.12 -m src.eval.run_benchmark
```

Use the benchmark output generated by the exact code/corpus/model configuration being reported. Record the run date and settings when sharing pass rate, faithfulness, citation accuracy, citation coverage, latency, or hallucination-related metrics. Do not describe an older result file as a result for newer code unless the benchmark was rerun.

### RAGAS-oriented evaluation

The repository also contains `eval_rag.py`. Check its configuration and ensure the expected FastAPI and Ollama services are available before running it. It may produce a CSV result file locally; generated output should be reviewed before deciding whether to commit it.

## Source Documents and Index Status

The raw-data folder currently contains HTML snapshots of the EU AI Act and GDPR. The source files should be treated separately from the processed index: downloading a new legislative version does not automatically update the chunks or vector/BM25 indexes.

- EU AI Act original text: [EUR-Lex, Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng)
- EU AI Act consolidated text dated 27 July 2026: [EUR-Lex consolidated text](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02024R1689-20260727)
- GDPR source: [EUR-Lex, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng)

## Data and Version Limitations

- The repository contains raw HTML sources, processed chunks, a BM25 index, and a local Qdrant database.
- A consolidated EU AI Act HTML source dated **27 July 2026** is present in `data/raw/`.
- **The existing processed index has not yet been rebuilt from that consolidated source.** Therefore, do not assume the current retrieval index includes all changes in that consolidated version.
- Deterministic extraction and exact-provision handling currently cover selected query patterns; they are not comprehensive legal rule coverage.
- Correct citation formatting does not itself prove the cited source entails the claim. Review the retrieved evidence and verification output for important questions.
- Redis/Valkey shared backend caching and cloud deployment are not included in the current implementation unless added in a later release.

## Troubleshooting

| Symptom | Checks |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run Uvicorn from the repository root using `src.api.main:app`. |
| UI cannot connect to FastAPI | Check that the API is running and that `API_BASE_URL` points to the right address. Inside Compose, the UI uses the API service name; on the host, use `http://localhost:8000`. |
| Ollama connection error in Docker | Confirm Ollama is running on the host, the model has been pulled, and the configured host URL is reachable from the container. |
| First startup appears slow | Wait for model initialization and check container logs before assuming the service has crashed. |
| Answer is flagged by NLI | Inspect the selected legal sources and claim audit. Do not lower verification thresholds just to make an answer pass. |

## Responsible Use

This is a research prototype and can produce incomplete, misattributed, or unsupported answers. It must not be used as the sole basis for legal decisions, regulatory filings, or compliance certification. Verify relevant claims against the official legal text and consult qualified legal professionals where appropriate.


## License

The original source code in this repository is licensed under the [MIT License](LICENSE).

The MIT License applies to the project's original code. Official EU legal materials, third-party libraries, and pretrained models remain subject to their respective reuse terms and licenses.

See the [`LICENSE`](LICENSE) file for the full license text.