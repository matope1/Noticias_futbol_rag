import streamlit as st
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("Falta GOOGLE_API_KEY en .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"

CHROMA_HOST = "chroma"   # nombre del servicio en docker-compose
CHROMA_PORT = 8000
COLLECTION_NAME = "deportes"

# ---------------------------------------------------------
# CONEXIÓN A CHROMA
# ---------------------------------------------------------
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(COLLECTION_NAME)

# ---------------------------------------------------------
# FUNCIONES RAG
# ---------------------------------------------------------
def embed(text):
    """Genera embeddings con Gemini."""
    r = genai.embed_content(model=EMBED_MODEL, content=text)
    return r["embedding"]

def retrieve(query):
    """Recupera contexto desde Chroma."""
    q_vec = embed(query)
    results = collection.query(query_embeddings=q_vec, n_results=5)
    docs = results["documents"][0]
    return "\n\n".join(docs)

def rag_answer(query):
    """Genera respuesta usando RAG (Chroma + Gemini)."""
    context = retrieve(query)

    if not context.strip():
        return "No encontré información relevante en la base vectorial."

    prompt = f"""
Responde SOLO usando el contexto. Si no está, dilo.

Pregunta:
{query}

Contexto:
{context}

Respuesta:
"""

    llm = genai.GenerativeModel(LLM_MODEL)
    return llm.generate_content(prompt).text

# ---------------------------------------------------------
# UI STREAMLIT (CHATBOT)
# ---------------------------------------------------------
st.set_page_config(page_title="Chat Deportivo RAG", layout="centered")
st.title("⚽🤖 Chat Deportivo RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Entrada del usuario
query = st.chat_input("Haz una pregunta sobre deportes españoles")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    answer = rag_answer(query)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
