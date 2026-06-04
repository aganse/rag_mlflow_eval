# rag_mlflow_eval

A small example project that builds a LangChain + FAISS retrieval pipeline, logs it to MLflow, and evaluates the resulting RAG app with MLflow GenAI scorers.

The default corpus is a small set of U.S. National Archives milestone documents covering the 13th, 14th, and 15th Amendments.

## What this repo does

- scrapes source documents from Archives.gov
- chunks and embeds them with OpenAI embeddings
- stores the chunks in a local FAISS index
- builds a LangChain retrieval QA chain
- logs the chain as an MLflow model
- creates or updates an MLflow evaluation dataset
- runs MLflow GenAI evaluation with LLM-as-a-judge scorers

## Repo layout

- `main.py` — end-to-end workflow
- `retriever_chain.py` — the logged LangChain retrieval model
- `utils.py` — document fetching, chunking, and FAISS creation
- `scorers.py` — prediction wrapper and MLflow scorer selection
- `smoke_test.py` — optional retriever-trace check for retrieval scorers
- `eval_dataset.py` — default small dataset and source URLs
- `eval_dataset_full.py` — larger alternative dataset
- `faiss_index/` — persisted vector index copied into the repo for model packaging

## Requirements

- Python 3.11+
- an OpenAI API key
- a running MLflow tracking server at `http://localhost:5000`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

Start MLflow locally in another terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

## Run

```bash
python main.py
```

On each run, the script will:

1. fetch the source documents
2. build a FAISS index in a temporary directory
3. copy that index into `faiss_index/`
4. log the retrieval chain to MLflow
5. load the logged model and run a sample prediction
6. create or merge an MLflow evaluation dataset
7. evaluate the model with MLflow GenAI scorers

## Evaluation behavior

By default, the project uses these scorers:

- `Correctness()`
- `RelevanceToQuery()`

If you enable `params["debug"] = True` in `main.py`, the repo also runs a smoke test that checks whether retriever spans were captured in MLflow traces. If that succeeds, it adds retrieval-specific scorers:

- `RetrievalRelevance()`
- `RetrievalGroundedness()`
- `RetrievalSufficiency()`

## Datasets

### Default dataset

`eval_dataset.py` uses a 3-document subset:

- 13th Amendment
- 14th Amendment
- 15th Amendment

### Larger dataset

`eval_dataset_full.py` contains a broader Civil War / Reconstruction-era evaluation set.

To switch datasets, update the imports in `main.py` from:

```python
from eval_dataset import get_dataset_name, get_url_listings, get_eval_records
```

to:

```python
from eval_dataset_full import get_dataset_name, get_url_listings, get_eval_records
```

## Notes

- `main.py` sets several environment variables up front to reduce FAISS / BLAS thread contention and MLflow worker noise on macOS.
- `retriever_chain.py` expects to find a packaged `faiss_index/` directory containing `index.faiss` and `index.pkl`.
- The current prompt is intentionally simple and keeps answers short.
- Source pages are scraped live from Archives.gov, so external site changes may affect retrieval quality.

## Typical workflow for experimentation

- change the source URLs or eval set
- adjust chunk size / overlap in `utils.create_faiss_database`
- change the model in `retriever_chain.py`
- re-run `python main.py`
- inspect traces, datasets, logged models, and eval results in MLflow

## License

MIT
