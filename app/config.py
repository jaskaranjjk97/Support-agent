import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "taskflow-help")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small

DATA_DIR = BASE_DIR / "data"
HELP_DOCS_DIR = DATA_DIR / "help_docs"
ACCOUNTS_FILE = DATA_DIR / "accounts.json"
ORDERS_FILE = DATA_DIR / "orders.json"

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
    )

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set.")
