# secure-ml-pipeline

A portfolio project demonstrating a **secure, automated ML pipeline** using CI/CD best practices, MLflow experiment tracking, and Docker containerization.

Built to showcase AI/ML DevSecOps skills: integrating security scanning directly into the ML workflow so that every code push is automatically audited, tested, and validated before a model artifact is produced.

---

## What This Pipeline Does

```
Push to main
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                 security-and-test job               │
│                                                     │
│  1. Install dependencies                            │
│  2. Bandit scan  ──── static code security check    │
│  3. pip-audit    ──── dependency vulnerability scan │
│  4. pytest       ──── model accuracy & output tests │
│  5. Train model  ──── logs metrics to MLflow        │
└─────────────────────────────────────────────────────┘
    │ (only if all above pass)
    ▼
┌─────────────────────────────────────────────────────┐
│                  docker-build job                   │
│                                                     │
│  Multi-stage build: train → minimal serving image   │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML model | scikit-learn (TF-IDF + Logistic Regression) |
| Experiment tracking | MLflow |
| Data handling | pandas, numpy |
| Testing | pytest |
| Static security scan | Bandit |
| Dependency audit | pip-audit |
| Containerisation | Docker (multi-stage) |
| CI/CD | GitHub Actions |

---

## Security Gates

Two automated security checks run on every push before any model is trained or image is built:

### 1. Bandit — Static Code Analysis
[Bandit](https://bandit.readthedocs.io/) scans Python source code for common security issues (e.g. hardcoded secrets, unsafe deserialization, shell injection).

```bash
bandit -r model/ -ll -ii
```
- `-ll` — report issues of **medium severity or higher**
- `-ii` — report issues of **medium confidence or higher**
- Fails the pipeline if any issues are found at this threshold

### 2. pip-audit — Dependency Vulnerability Scan
[pip-audit](https://pypi.org/project/pip-audit/) checks all pinned dependencies against the [OSV](https://osv.dev/) vulnerability database.

```bash
pip-audit --requirement requirements.txt
```
- Fails the pipeline if any known CVEs are found in the dependency tree
- Reports are uploaded as CI artifacts for review

---

## Model

A binary spam/fraud classifier built as a **scikit-learn Pipeline**:

1. `TfidfVectorizer` — converts raw text to TF-IDF feature vectors (bigrams, top 500 features)
2. `LogisticRegression` — classifies as spam (`1`) or ham (`0`)

Trained on a synthetic dataset of spam and legitimate messages for reproducible CI runs.

**Metrics logged to MLflow:** accuracy, precision, recall, F1.

**Minimum quality gates enforced by pytest:**
- Accuracy ≥ 70%
- F1 score ≥ 60%
- All predictions within `{0, 1}`
- Probabilities sum to 1.0

---

## Project Structure

```
secure-ml-pipeline/
├── .github/
│   └── workflows/
│       └── pipeline.yml      # GitHub Actions CI/CD workflow
├── model/
│   └── train.py              # Training script with MLflow logging
├── tests/
│   └── test_model.py         # pytest: accuracy, shape, output, artifact
├── Dockerfile                # Multi-stage: trainer → minimal server
├── requirements.txt          # Pinned Python dependencies
└── README.md
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- Docker (optional, for container build)

### Setup

```bash
git clone https://github.com/LoryGlory/secure-ml-pipeline.git
cd secure-ml-pipeline

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train the model

```bash
python model/train.py
```

Metrics are logged to `mlruns/`. View them in the MLflow UI:

```bash
mlflow ui
# open http://localhost:5000
```

### Run tests

```bash
pytest tests/ -v
```

### Run security scans

```bash
# Static code analysis
bandit -r model/ -ll -ii

# Dependency vulnerability audit
pip-audit --requirement requirements.txt
```

### Build and run Docker image

```bash
docker build -t secure-ml-pipeline .

# Classify a message
docker run secure-ml-pipeline "Win a free iPhone now!!!"
# Output: 'Win a free iPhone now!!!' -> SPAM
```

---

## CI/CD Pipeline (GitHub Actions)

The workflow at [.github/workflows/pipeline.yml](.github/workflows/pipeline.yml) runs on every push and pull request to `main`.

**Job 1 — `security-and-test`** (must pass before Docker build):
1. Install pinned dependencies
2. Bandit static security scan → uploads `bandit-report.json` artifact
3. pip-audit dependency scan → uploads `pip-audit-report.json` artifact
4. pytest model tests
5. Train model → uploads `mlruns/` and `spam_classifier.pkl` artifacts

**Job 2 — `docker-build`** (runs only if Job 1 passes):
1. Multi-stage Docker build (train inside builder, copy artifact to slim server image)
2. Tagged as `secure-ml-pipeline:<sha>` and `secure-ml-pipeline:latest`

---

## License

MIT
