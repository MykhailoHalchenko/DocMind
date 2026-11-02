FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data \
    && useradd --create-home --uid 10001 docmind \
    && chown -R docmind:docmind /app

ENV FAISS_INDEX_PATH=/app/data/knowledge_base.faiss \
    FAISS_METADATA_PATH=/app/data/knowledge_base.json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER docmind

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "main.py", "server", "--no-browser", "--host", "0.0.0.0", "--port", "8000"]
