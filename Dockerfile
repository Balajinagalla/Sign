# Dockerfile for ISL Recognition System
# Usage:
#   docker build -t isl-recognition .
#   docker run -p 8000:8000 isl-recognition           # API server
#   docker run -p 8501:8501 isl-recognition streamlit  # Streamlit app

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    espeak \
    libespeak-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements_full.txt .
RUN pip install --no-cache-dir -r requirements_full.txt

# Copy application files
COPY *.py ./
COPY best.pt* ./
COPY best.onnx* ./
COPY sign_references/ ./sign_references/
COPY Data/data.yaml ./Data/data.yaml

# Expose ports (API: 8000, Streamlit: 8501)
EXPOSE 8000 8501

# Default: start API server
# Override with: docker run ... streamlit
ENTRYPOINT ["python"]
CMD ["api_server.py"]
