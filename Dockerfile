FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y artefactos
COPY app.py .
COPY artifacts_regresion/ ./artifacts_regresion/

# Juan Camilo: descomenta cuando tengas tus artefactos
# COPY artifacts_clasificacion/ ./artifacts_clasificacion/

EXPOSE 8050

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "2", "--timeout", "120"]
