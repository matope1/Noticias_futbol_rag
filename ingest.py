import logging
from pathlib import Path
from datetime import datetime

import chromadb
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    GOOGLE_API_KEY,
    EMBED_MODEL,
    COLLECTION_NAME,
    CHROMA_HOST,
    CHROMA_PORT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# CONFIGURACIÓN
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PDF_PATH = Path("ingesta/documentos/Noticias_Futbol.pdf")

genai.configure(api_key=GOOGLE_API_KEY)


# VALIDACIÓN DEL PDF
if not PDF_PATH.exists():
    raise FileNotFoundError(f"❌ No se encontró el archivo PDF en: {PDF_PATH}")

logging.info(f"📄 Cargando PDF: {PDF_PATH}")


# CARGA DEL PDF
loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()
logging.info(f"📄 PDF cargado correctamente ({len(pages)} páginas)")


# CHUNKING
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
    is_separator_regex=True,
)

docs = splitter.split_documents(pages)
logging.info(f"✂️ Chunks generados: {len(docs)}")


# FUNCIÓN DE EMBEDDINGS
def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []

    for text in texts:
        response = genai.embed_content(
            model=EMBED_MODEL,
            content=text
        )
        vectors.append(response["embedding"])

    return vectors


# ENRIQUECIMIENTO DE METADATOS
def infer_category(text: str) -> str:
    text_lower = text.lower()

    if any(word in text_lower for word in ["real madrid", "barcelona", "liga", "fútbol", "futbol"]):
        return "futbol"
    if any(word in text_lower for word in ["baloncesto", "nba", "acb"]):
        return "baloncesto"
    if any(word in text_lower for word in ["tenis", "roland garros", "wimbledon"]):
        return "tenis"

    return "deporte_general"


# CONEXIÓN A CHROMA
logging.info(f"🔌 Conectando a ChromaDB en {CHROMA_HOST}:{CHROMA_PORT}...")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

try:
    client.delete_collection(COLLECTION_NAME)
    logging.info(f"🗑 Colección '{COLLECTION_NAME}' eliminada.")
except Exception:
    logging.info(f"ℹ️ La colección '{COLLECTION_NAME}' no existía.")

collection = client.create_collection(name=COLLECTION_NAME)
logging.info(f"📦 Colección '{COLLECTION_NAME}' creada.")


# PREPARAR DATOS
texts = [doc.page_content for doc in docs]
ids = [f"{PDF_PATH.stem}_chunk_{i}" for i in range(len(texts))]

metadatas = []
for i, doc in enumerate(docs):
    chunk_text = doc.page_content

    metadatas.append({
        "source": str(PDF_PATH),
        "page": doc.metadata.get("page", -1),
        "chunk_id": i,
        "category": infer_category(chunk_text),
        "char_count": len(chunk_text),
        "ingestion_date": datetime.now().isoformat()
    })


# GENERAR EMBEDDINGS
logging.info("🧮 Generando embeddings con Gemini...")
vectors = embed_texts(texts)


# INSERTAR EN CHROMA
logging.info("📥 Insertando documentos en Chroma...")
collection.add(
    ids=ids,
    documents=texts,
    embeddings=vectors,
    metadatas=metadatas
)

logging.info("✅ Ingesta completada correctamente.")