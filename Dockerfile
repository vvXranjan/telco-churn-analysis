FROM python:3.13-slim

WORKDIR /app

# Minimal runtime deps for the inference API (see requirements-app.txt).
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app ./app
COPY models ./models

# The application loads the persisted model artifact at startup; it never retrains.
# PORT is respected as an environment variable (default 8000).
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','8000')+'/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]