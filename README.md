# rag_mlflow_eval

This project evaluates question-answering (QA) behavior with MLflow,
including both a retrieval-augmented generation (RAG) mode and a no-RAG
baseline mode for performance comparison.

The default corpus is a set of space-related news items that happened in
2024-2026, after the knowledge cutoff of OpenAI's gpt-4o-mini LLM model
which is used as the core of the RAG model.  So we can check the drop
in correctness by turning off the RAG and letting the model hallucinate
answers to events that occurred more recently than its training data.

Note to MLflow a "dataset" is just the set of test input queries and their
expected answers, but the dataset modules in this repo additionally contain
the list of URLs of associated documents to use for the RAG.


## What this repo does

- loads an evaluation dataset definition from `datasets/`
- optionally scrapes source documents from webpages listed with dataset
- optionally chunks and embeds them with OpenAI embeddings
- optionally stores the chunks in a local FAISS index
- builds either a RAG chain or a no-RAG QA chain
- logs the chain as an MLflow model
- creates or updates an MLflow evaluation dataset
- runs MLflow GenAI evaluation with LLM-as-a-judge scorers
- logs the selected workflow parameters to MLflow params


## Requirements

- Python 3.11+
- an OpenAI API key
- a running MLflow tracking server at `http://localhost:5000`
  (see [docker_mlflow_db](https://github.com/aganse/docker_mlflow_db)
  or start one in another terminal via
  `mlflow server --host 127.0.0.1 --port 5000`)

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


## Configuration

Edit the `params` dictionary near the top of `main.py`:

```python
params = {
    "experiment_name": "Space News RAG",
    "mlflow_tracking_uri": "http://localhost:5000",
    "mode": "rag",                                # "rag" or "no_rag"
    "verbose": False,
    "retrieval_backend": "faiss",                 # (for now the only choice)
    "dataset": "space_news",                      # "space_news", "civil_war_16", "testA" from datasets dir
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retrieval_top_k": 5,
    "embedding_model": "text-embedding-3-small",  # langchain default is "text-embedding-ada-002"
    "base_llm": "gpt-4o-mini",                    # the arbitrary model the RAG is built around
    "judge_llm": "openai:/gpt-4o-mini",           # mlflow default, note "openai:/" is required
}
```

For `embedding_model` and `judge_llm` one can set `""` (or omit the key from
params) to fall back to MLflow's default model for each.
See [this MLflow page](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/supported-models/#direct-model-providers)
for a list of model provider names and formatting for MLflow's LLM-judges.
See [this OpenAI page](https://developers.openai.com/api/docs/guides/embeddings)
for a list of OpenAI embedding models (and prices) available via API.
The `chunk_size`, `chunk_overlap`, `retrieval_top_k` params are validated and
logged on every run but are only actually used when `mode == "rag"`.


## Run

Edit `params` at top of `main.py`, ensure you're in the python environment for
this project, and then run `python main.py`.

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

- a model logging run such as `log_qa_rag_faiss_testA` in which the RAG agent
  is created using the url_listings associated with the given dataset.
- an evaluation run such as `eval_qa_rag_faiss_testA_baseline` in which the
  RAG agent's performance is evaluated on the dataset's eval Q&A set.

In the MLflow UI, both appear in the general runs list, while only the
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


## Retrieval backend notes

The current retrieval implementation uses LangChain's FAISS integration.

A future pgvector option is intentionally not implemented yet, but design notes
for that later addition are captured in `notes.pgvector.backend.txt`.

## Notes

- `main.py` sets several environment variables up front to reduce FAISS / BLAS
  thread contention and MLflow worker noise on macOS.
- `retriever_chain.py` expects a packaged `faiss_index/` directory when running
  in RAG mode.
- Source pages are scraped live from webpages (especially news and wikipedia);
  so note external site changes may affect retrieval quality.
