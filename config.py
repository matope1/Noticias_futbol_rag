import os
from dotenv import load_dotenv

# ---------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("❌ Falta GOOGLE_API_KEY en el archivo .env")

# Modelos Gemini
EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"

# ChromaDB
COLLECTION_NAME = "deportes"
CHROMA_HOST = "chroma"
CHROMA_PORT = 8000

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
TOP_K_RESULTS = 5