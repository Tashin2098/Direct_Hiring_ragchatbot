import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Core
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment/.env")

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Indexing / retrieval
DATA_DIR = os.getenv("DATA_DIR", "data")
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "3"))
EMBED_RETRY_BACKOFF = float(os.getenv("EMBED_RETRY_BACKOFF", "2.0"))

# Ensure dirs exist
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
