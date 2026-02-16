# Noticias_futbol_rag

## Indicacion para ejecutar la aplicacion

### 1. Clonar repositorio
```bash
git clone https://github.com/matope1/Noticias_futbol_rag
```
```bash
cd Noticias_futbol_rag
```
### 2. Editar .env
- Cambiar "tu_api_key" por una key valida de gemini

### 3. Asegurarse que docker esta abierto:
  - Levantar contenedores
```bash
docker compose up -d
```
### 4. Hacer ingesta de documentos
```bash
uv run ingesta/ingest.py
```
### 5. Acceder a streamlit:
```bash
http://localhost:8501/
```
### - Bateria de preguntas para hacer a el chatbot:
```bash
¿Quién va primero en la clasificación?
```
```bash
¿Cuántos puntos tiene el Real Madrid?
```
```bash
¿Cuántos partidos ha ganado el FC Barcelona?
```
```bash
¿Cómo está la lucha por el ascenso en Segunda División?
```
```bash
¿Qué se dice sobre la permanencia en la parte baja de la tabla?
```
```bash
¿Qué destaca del juego del Barcelona esta temporada?
```
