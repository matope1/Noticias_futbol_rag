import streamlit as st
from rag_service import rag_answer


# CONFIGURACIÓN DE STREAMLIT
st.set_page_config(page_title="Chat Deportivo RAG", layout="centered")


# CABECERA
st.title("⚽🤖 Chat Deportivo RAG")
st.caption("Haz preguntas sobre deportes españoles usando Chroma + Gemini")


# ESTADO DE SESIÓN
if "messages" not in st.session_state:
    st.session_state.messages = []


# MOSTRAR HISTORIAL
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ENTRADA DEL USUARIO
query = st.chat_input("Haz una pregunta sobre deportes españoles")

if query:
    # Guardar mensaje del usuario
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando respuesta..."):
            try:
                result = rag_answer(query)
                answer = result["answer"]
                context = result["context"]
                sources = result["sources"]

                st.write(answer)

                with st.expander("Ver contexto recuperado"):
                    if context:
                        st.text(context)
                    else:
                        st.write("No se recuperó contexto.")

                with st.expander("Ver fuentes"):
                    metadatas = sources.get("metadatas", [])
                    distances = sources.get("distances", [])
                    ids = sources.get("ids", [])

                    if metadatas:
                        for i, meta in enumerate(metadatas):
                            distance = distances[i] if i < len(distances) else "N/A"
                            chunk_id = ids[i] if i < len(ids) else "N/A"

                            st.write(
                                f"**Fragmento {i+1}** | "
                                f"id={chunk_id} | "
                                f"source={meta.get('source')} | "
                                f"page={meta.get('page')} | "
                                f"category={meta.get('category')} | "
                                f"distance={distance}"
                            )
                    else:
                        st.write("No hay metadatos disponibles.")

            except Exception as e:
                answer = f"Ocurrió un error al procesar la consulta: {e}"
                st.error(answer)

    # Guardar respuesta
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })