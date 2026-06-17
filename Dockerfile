FROM python:3.11-slim

WORKDIR /app

# libgomp1 = GNU OpenMP runtime. scikit-learn's tree-based estimators,
# xgboost, and lightgbm are all compiled against it for parallelism, and
# unpickling them dlopen()s it at import time. It's an OS-level shared
# library, not something `pip install` can provide — without it you get
# "libgomp.so.1: cannot open shared object file" when joblib.load() touches
# any of those models. Placed before COPY so this layer caches independently
# of source/model changes.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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