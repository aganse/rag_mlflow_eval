# rag_mlflow_eval

A small example project that evaluates question-answering behavior with MLflow,
including both a retrieval-augmented generation (RAG) mode and a no-RAG
baseline mode for performance comparison.

The default corpus is a set of U.S. National Archives milestone documents defined
by the URLs in the modules in datasets/ and pointed to by the "dataset" parameter,
then scraped from those webpages at runtime.


## What this repo does

- loads an evaluation dataset definition from `datasets/`
- optionally scrapes source documents from Archives.gov
- optionally chunks and embeds them with OpenAI embeddings
- optionally stores the chunks in a local FAISS index
- builds either a RAG chain or a no-RAG QA chain
- logs the chain as an MLflow model
- creates or updates an MLflow evaluation dataset
- runs MLflow GenAI evaluation with LLM-as-a-judge scorers
- logs the selected workflow parameters to MLflow params

## Repo layout

- `main.py` — end-to-end workflow orchestration and config
- `chain_factory.py` — shared QA model construction helpers
- `retriever_chain.py` — logged MLflow wrapper for RAG mode
- `no_rag_chain.py` — logged MLflow wrapper for no-RAG mode
- `logged_model_config.py` — packaged runtime config for the logged model
- `retrieval_backends/` — retrieval backend helpers (`faiss` today)
- `datasets/` — dataset registry plus individual dataset definition files
- `utils.py` — document fetching, chunking, and FAISS creation
- `scorers.py` — prediction wrapper and MLflow scorer selection
- `smoke_test.py` — retriever-trace check for retrieval scorers in RAG mode
- `faiss_index/` — persisted vector index copied into the repo for packaging
- `notes.pgvector.backend.txt` — notes for a future pgvector backend option

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

Start MLflow locally in another terminal (or of course you can use an existing one):

```bash
mlflow server --host 127.0.0.1 --port 5000
```

## Configuration

Edit the `params` dictionary near the top of `main.py`:

```python
params = {
    "experiment_name": "History docs RAG",
    "mlflow_tracking_uri": "http://localhost:5000",
    "mode": "rag",                # "rag" | "no_rag"
    "verbose": False,
    "retrieval_backend": "faiss", # future-ready; only "faiss" today
    "dataset": "testA",           # "testA" | "civil_war_16"
    "base_llm": "gpt-4.1-mini",
    "judge_llm": "",             # "" -> use MLflow default judge model
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retrieval_top_k": 4,
}
```

### Parameter notes

- `mode`
  - `"rag"` builds retrieval artifacts, logs a retrieval chain, and attempts
    retrieval-specific evaluation.
  - `"no_rag"` skips retrieval setup and logs a closed-book QA baseline.
- `verbose`
  - Enables lightweight progress prints.
  - It does not dump full traces or other large artifacts.
- `retrieval_backend`
  - Currently only `"faiss"` is implemented.
  - The structure is designed so a future `"pgvector"` backend can be added.
- `dataset`
  - Uses a short dataset name that maps to a module in `datasets/`.
  - Each dataset module also provides its own long-form MLflow dataset name via
    `get_dataset_name()`.
- `base_llm`
  - Controls the answering model used in both `rag` and `no_rag` modes.
- `judge_llm`
  - Controls the MLflow LLM-as-a-judge model used by the evaluation scorers.
  - Use `""` (or omit the key) to fall back to MLflow's default judge model.
  - When set, use MLflow judge model format such as `openai:/gpt-4o-mini`.
- `chunk_size`, `chunk_overlap`, `retrieval_top_k`
  - These are validated and logged on every run.
  - They are only used when `mode == "rag"`.

The logged model name is created automatically from the high-level app variant:

- `qa_rag_faiss_testA`
- `qa_no_rag_testA`

The evaluation run name is also created automatically, using a baseline label:

- `eval_qa_rag_faiss_testA_baseline`
- `eval_qa_no_rag_testA_baseline`

Detailed settings such as `base_llm`, `judge_llm`, chunking, and retrieval top-k
are logged as MLflow params instead of being embedded in the model name.

## Run

```bash
python main.py
```

On each run, the script will:

1. validate the configured parameters
2. load the selected dataset module from `datasets/`
3. if `mode == "rag"`, build a FAISS index from the dataset source URLs
4. write a small packaged runtime config for the logged model
5. log either the RAG model or the no-RAG model to MLflow
6. load the logged model and run a sample prediction
7. create or merge an MLflow evaluation dataset
8. run MLflow GenAI evaluation as a separate evaluation run
9. log the configured workflow parameters to the model logging run

This means a typical `main.py` execution creates two MLflow run entries:

- a model logging run such as `log_qa_rag_faiss_testA`
- a named evaluation run such as `eval_qa_rag_faiss_testA_baseline`

Both runs log the core workflow parameters. The evaluation run additionally logs
`evaluation_label=baseline`.

In the MLflow UI, both may appear in the general runs list, while only the
evaluation run appears in the dedicated Evaluation Runs view.

## Evaluation behavior

By default, the project always uses these scorers:

- `Correctness()`
- `RelevanceToQuery()`

When `mode == "rag"`, the project also runs a retrieval smoke test to confirm
that retriever spans were captured in MLflow traces. If that succeeds, it adds
these retrieval-specific scorers:

- `RetrievalRelevance()`
- `RetrievalGroundedness()`
- `RetrievalSufficiency()`

In `no_rag` mode, retrieval-specific scorers are skipped automatically.

## Datasets

Datasets live in the `datasets/` package.

### Current dataset short names

- `testA`
  - module: `datasets/testA.py`
  - current long MLflow dataset name:
    `usmilestonedocs_civilwar_3docsubset`
- `civil_war_16`
  - module: `datasets/civil_war_16.py`
  - current long MLflow dataset name:
    `usmilestonedocs_civilwar_all16`

To add another dataset later:

1. create a new module in `datasets/`
2. implement:
   - `get_dataset_name()`
   - `get_url_listings()`
   - `get_eval_records()`
3. register the module in `datasets/__init__.py`
4. set `params["dataset"]` to the new short name

## Retrieval backend notes

The current retrieval implementation uses LangChain's FAISS integration.

A future pgvector option is intentionally not implemented yet, but design notes
for that later addition are captured in `notes.pgvector.backend.txt`.

## Notes

- `main.py` sets several environment variables up front to reduce FAISS / BLAS
  thread contention and MLflow worker noise on macOS.
- `retriever_chain.py` expects a packaged `faiss_index/` directory when running
  in RAG mode.
- The current prompts are intentionally simple and keep answers short.
- Source pages are scraped live from Archives.gov, so external site changes may
  affect retrieval quality.

## Typical workflow for experimentation

- switch between `rag` and `no_rag`
- switch datasets by changing `params["dataset"]`
- adjust `base_llm`, `judge_llm`, `chunk_size`, `chunk_overlap`, or `retrieval_top_k`
- re-run `python main.py`
- inspect traces, datasets, logged models, eval runs, eval results, and logged
  params in MLflow

## License

MIT
