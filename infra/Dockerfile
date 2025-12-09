# Build stage
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Create requirements file
RUN echo "scikit-learn==1.3.2" > requirements-prod.txt && \
    echo "pandas==2.1.4" >> requirements-prod.txt && \
    echo "numpy==1.26.3" >> requirements-prod.txt && \
    echo "joblib==1.3.2" >> requirements-prod.txt && \
    echo "fastapi==0.109.0" >> requirements-prod.txt && \
    echo "uvicorn[standard]==0.27.0" >> requirements-prod.txt && \
    echo "pydantic==2.5.3" >> requirements-prod.txt

RUN pip install --no-cache-dir -r requirements-prod.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application files
COPY src/api.py /app/src/api.py
COPY models/model.pkl /app/models/model.pkl

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
