FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .

# create a non root user and switch to it
RUN useradd -create-home --uid 1000 appuser

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# create directories for app data and huggingface cache, and set ownership to the non-root user
RUN mkdir -p /app/data /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

# swithc user context to non root user
USER appuser

EXPOSE 8000

# Application health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]