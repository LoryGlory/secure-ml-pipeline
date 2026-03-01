# Multi-stage build: train → serve
# Stage 1: Train the model
FROM python:3.11-slim AS trainer

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ ./model/

RUN python model/train.py

# Stage 2: Minimal serving image
FROM python:3.11-slim AS server

WORKDIR /app

# Only install runtime deps (no dev/test/security tools in prod image)
RUN pip install --no-cache-dir \
    scikit-learn==1.4.2 \
    mlflow==2.12.2 \
    pandas==2.2.2 \
    numpy==1.26.4

# Copy trained artifact from trainer stage
COPY --from=trainer /app/model/spam_classifier.pkl ./model/spam_classifier.pkl
COPY model/train.py ./model/train.py

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Serve via MLflow models serve or a simple prediction script
# Expose MLflow UI port
EXPOSE 5000

CMD ["python", "-c", "\
import pickle, sys; \
model = pickle.load(open('model/spam_classifier.pkl', 'rb')); \
texts = sys.argv[1:] if len(sys.argv) > 1 else ['Enter text as argument']; \
preds = model.predict(texts); \
[print(f'{t!r} -> {\"SPAM\" if p else \"HAM\"}') for t, p in zip(texts, preds)] \
"]
