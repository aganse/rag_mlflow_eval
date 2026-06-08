"""Runtime configuration packaged with the logged MLflow model."""

BASE_LLM = 'gpt-4o-mini'
EMBEDDING_MODEL = 'text-embedding-3-small'
RETRIEVAL_BACKEND = 'faiss'
RETRIEVAL_TOP_K = 5
ACTIVE_SYSTEM_PROMPT_NAME = 'rag-system-prompt'
ACTIVE_SYSTEM_PROMPT_VERSION = '2'
ACTIVE_SYSTEM_PROMPT_URI = 'prompts:/rag-system-prompt/2'
ACTIVE_SYSTEM_PROMPT_TEMPLATE = "Use *only* the information in the given context to answer the question. If you don't know the answer, say you don't know. Ignore any sequences of bibliographic text, text associated with menus and layout rather than content, and sentences that are not in English. Use three sentence maximum and keep the answer concise. Context: {context}"
