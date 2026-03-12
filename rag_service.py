import chromadb
import google.generativeai as genai

from config import (
    GOOGLE_API_KEY,
    EMBED_MODEL,
    LLM_MODEL,
    COLLECTION_NAME,
    CHROMA_HOST,
    CHROMA_PORT,
    TOP_K_RESULTS,
)


# CONFIGURACIÓN GEMINI
genai.configure(api_key=GOOGLE_API_KEY)


# CONEXIÓN A CHROMA
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(COLLECTION_NAME)


# MODELO GENERATIVO
llm = genai.GenerativeModel(LLM_MODEL)


# FUNCIONES RAG
def embed_query(text: str) -> list[float]:
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )
    return response["embedding"]


def retrieve_documents(query: str, n_results: int = TOP_K_RESULTS) -> dict:
    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )

    return {
        "documents": results.get("documents", [[]])[0],
        "metadatas": results.get("metadatas", [[]])[0],
        "distances": results.get("distances", [[]])[0],
        "ids": results.get("ids", [[]])[0],
    }


def build_context(retrieved: dict) -> str:
    docs = retrieved.get("documents", [])

    if not docs:
        return ""

    fragments = []
    for i, doc in enumerate(docs, start=1):
        fragments.append(f"[Fragmento {i}]\n{doc}")

    return "\n\n".join(fragments)


def build_prompt(query: str, context: str) -> str:
    return f"""
Eres un asistente de preguntas y respuestas sobre deportes españoles.

Reglas:
- Responde SOLO usando el contexto.
- No inventes información.
- Si la respuesta no aparece en el contexto, responde exactamente:
  "No encontré esa información en la base vectorial."
- Sé claro y preciso.

Pregunta:
{query}

Contexto:
{context}

Respuesta:
""".strip()


def generate_answer(prompt: str) -> str:
    response = llm.generate_content(prompt)

    if hasattr(response, "text") and response.text:
        return response.text.strip()

    return "No pude generar una respuesta válida."


def rag_answer(query: str) -> dict:
    retrieved = retrieve_documents(query)
    context = build_context(retrieved)

    if not context.strip():
        return {
            "answer": "No encontré información relevante en la base vectorial.",
            "context": "",
            "sources": retrieved
        }

    prompt = build_prompt(query, context)
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "context": context,
        "sources": retrieved
    }