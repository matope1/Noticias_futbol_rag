FROM python:3.11-slim

WORKDIR /app

# Copia requirements
COPY requirements.txt .

# Instala dependencias
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia tu código
COPY app.py .

# Exponer puerto para Streamlit
EXPOSE 8501

# Ejecutar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
