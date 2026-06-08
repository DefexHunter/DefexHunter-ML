FROM python:3.11-slim

WORKDIR /app

# install deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source and pre-trained models
COPY src/ src/
COPY models/ models/

# HuggingFace Spaces expects port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]