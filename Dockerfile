# Car Price Prediction - production image
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; scikit-learn wheels are self-contained.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY data/raw/ data/raw/

# Train the model at image build time so the container is self-contained and
# reproducible from a clean image with no external volume required.
RUN python scripts/train_model.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
