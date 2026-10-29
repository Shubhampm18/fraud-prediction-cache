FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /service

RUN pip install --no-cache-dir \
    fastapi==0.162.0 \
    "uvicorn[standard]==0.54.0" \
    numpy==2.5.4 \
    pandas==3.1.0 \
    scikit-learn==1.9.1 \
    joblib==1.6.0 \
    cachetools==7.2.7

COPY app ./app
COPY models ./models

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]