import logging
from pathlib import Path
import os
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

load_dotenv()  # Carga variables del archivo .env

PDF_PATH = Path("ingesta\\documentos\\Noticias_Futbol.pdf")
COLLECTION_NAME = "deportes"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ Falta GEMINI_API_KEY en el archivo .env")

genai.configure(api_key=GEMINI_API_KEY)

# Modelo de embeddings de Gemini
EMBED_MODEL = "models/text-embedding-004"

# ---------------------------------------------------------
# VALIDACIÓN DEL PDF
# ---------------------------------------------------------
if not PDF_PATH.exists():
    raise FileNotFoundError(f"❌ No se encontró el archivo PDF en: {PDF_PATH}")

logging.info(f"📄 Cargando PDF: {PDF_PATH}")

# ---------------------------------------------------------
# CARGA DEL PDF
# ---------------------------------------------------------
loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()
logging.info(f"📄 PDF cargado correctamente ({len(pages)} páginas)")

# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

docs = splitter.split_documents(pages)
logging.info(f"✂️ Chunks generados: {len(docs)}")

# ---------------------------------------------------------
# FUNCIÓN: EMBEDDINGS CON GEMINI
# ---------------------------------------------------------
def embed_texts(texts):
    """Genera embeddings usando Gemini gemini-embedding-001."""
    vectors = []
    for t in texts:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=t
        )
        vectors.append(response["embedding"])
    return vectors


# ---------------------------------------------------------
# CONEXIÓN A CHROMA
# ---------------------------------------------------------
logging.info(f"🔌 Conectando a ChromaDB en {CHROMA_HOST}:{CHROMA_PORT}...")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Crear colección limpia
try:
    client.delete_collection(COLLECTION_NAME)
    logging.info(f"🗑 Colección '{COLLECTION_NAME}' eliminada (si existía).")
except Exception:
    logging.info(f"ℹ️ La colección '{COLLECTION_NAME}' no existía.")

collection = client.create_collection(name=COLLECTION_NAME)
logging.info(f"📦 Colección '{COLLECTION_NAME}' creada.")

# ---------------------------------------------------------
# PREPARAR DATOS
# ---------------------------------------------------------
texts = [doc.page_content for doc in docs]
ids = [f"{PDF_PATH.stem}_chunk_{i}" for i in range(len(texts))]

logging.info("🧮 Generando embeddings con Gemini...")
vectors = embed_texts(texts)

# ---------------------------------------------------------
# INSERTAR EN CHROMA
# ---------------------------------------------------------
logging.info("📥 Insertando documentos en Chroma...")
collection.add(
    documents=texts,
    ids=ids,
    embeddings=vectors,
    metadatas=[{"source": str(PDF_PATH)} for _ in texts]
)

logging.info("✅ Ingesta completada correctamente.")

