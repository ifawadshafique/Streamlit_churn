# Multi-purpose image: runs the FastAPI service by default.
# Override CMD to run the Streamlit dashboard instead (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

# System deps for lightgbm/xgboost/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build-time: generate data and train the model so the image is
# self-contained and ready to serve immediately.
RUN python src/generate_data.py \
    && python src/preprocessing.py \
    && python src/feature_engineering.py \
    && python src/train.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
