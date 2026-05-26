#Dockerfile del Taller 12
#PROYECTO 2 LA TUPLA ACTD

# syntax=docker/dockerfile:1 
#FROM ubuntu:22.04
FROM python:3.10-slim
# install app dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY artifacts_regresion ./artifacts_regresion
COPY artifacts_clasificacion ./artifacts_clasificacion
COPY assets ./assets
EXPOSE 8050
# install app



# final configuration
CMD ["python3","app.py"]
