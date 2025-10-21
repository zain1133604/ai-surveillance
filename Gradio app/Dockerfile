# ===== BASE IMAGE =====
FROM python:3.10-slim

# ===== SETUP WORKDIR =====
RUN mkdir -p /app/runs && chmod -R 777 /app/runs
WORKDIR /app

# ===== INSTALL SYSTEM DEPENDENCIES (CRITICAL FIXES HERE) =====
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        git \
        libhdf5-dev \
        libblas-dev \
        liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# ===== COPY PROJECT FILES =====
COPY requirements.txt . 
COPY . .

# ===== INSTALL PYTHON DEPENDENCIES =====
RUN pip install --upgrade pip && \
    pip install --no-cache-dir tensorflow h5py && \
    pip install --no-cache-dir -r requirements.txt

# ===== FIX PERMISSION + CACHE ERRORS (THE FINAL FIX IS HERE) =====
# 1. FORCE HOME DIR TO /app (Fixes keras-facenet PermissionError)
ENV HOME=/app

# 2. Redirect other caches to the writable /app directory
ENV KERAS_HOME="/app/.keras"
ENV MPLCONFIGDIR="/app/.matplotlib"
ENV TORCH_HOME="/app/.cache/torch"
ENV YOLO_CONFIG_DIR="/app/.config/Ultralytics"
ENV FACE_ALIGNMENT_CACHE="/app/.cache/face_alignment"
ENV XDG_CACHE_HOME="/app/.cache"

# ===== CREATE DIRECTORIES AND GIVE PERMISSIONS =====
# Ensure all cache dirs are created and writable
RUN mkdir -p /app/.keras /app/.cache/torch /app/.matplotlib /app/.config/Ultralytics /app/.cache/face_alignment && \
    chmod -R 777 /app/.cache /app/.matplotlib /app/.config /app/.keras /app

# ===== ENVIRONMENT VARIABLES =====
# Ensure Gradio binds correctly inside containers (Docker or Hugging Face)
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SHARE=True

# ===== EXPOSE PORT =====
EXPOSE 7860

# ===== RUN GRADIO APP =====
CMD ["python", "app.py"]