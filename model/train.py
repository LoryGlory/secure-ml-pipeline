"""
Train a spam/fraud classifier and log experiment to MLflow.

Uses the UCI SMS Spam Collection dataset (synthetic fallback for CI).
Outputs: trained model artifact saved to model/spam_classifier.pkl
"""

import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_classifier.pkl")

# Minimal synthetic spam dataset for reproducible CI runs
SYNTHETIC_DATA = [
    ("Free money! Click now to claim your prize!!!", 1),
    ("Win a free iPhone today! Limited time offer!", 1),
    ("URGENT: Your account has been compromised. Verify now.", 1),
    ("Congratulations! You've been selected for a cash reward.", 1),
    ("Get rich quick with this one weird trick!", 1),
    ("Buy cheap meds online, no prescription needed.", 1),
    ("You are a winner! Claim your $1000 gift card.", 1),
    ("Meet singles in your area tonight!", 1),
    ("MAKE MONEY FAST working from home!!", 1),
    ("Your loan has been approved. Click to receive funds.", 1),
    ("Hey, are we still on for lunch tomorrow?", 0),
    ("Please find the meeting notes attached.", 0),
    ("Your order has been shipped and will arrive Thursday.", 0),
    ("Can you review the pull request when you have time?", 0),
    ("Happy birthday! Hope you have a wonderful day.", 0),
    ("The project deadline has been moved to next Friday.", 0),
    ("I'll be late to the standup, starting without me.", 0),
    ("Thanks for the help earlier, really appreciate it.", 0),
    ("Reminder: team retrospective at 3pm today.", 0),
    ("Let me know if you need anything else from my end.", 0),
    ("Your package has been delivered to the front door.", 0),
    ("The report is ready for your review in the shared drive.", 0),
    ("Can we reschedule our 1:1 to Thursday?", 0),
    ("Just checking in to see how the onboarding is going.", 0),
    ("Your subscription renewal is coming up next month.", 0),
]


def load_data():
    """Load synthetic spam/ham dataset as a DataFrame."""
    df = pd.DataFrame(SYNTHETIC_DATA, columns=["text", "label"])
    return df


def build_pipeline():
    """Build a TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def train():
    """Train the classifier, log metrics to MLflow, save artifact."""
    df = load_data()
    X, y = df["text"], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_experiment("spam-classifier")

    with mlflow.start_run():
        model = build_pipeline()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "max_features": 500,
            "ngram_range": "(1,2)",
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Training complete.")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save local artifact for serving / testing
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {MODEL_PATH}")

    return model, metrics


if __name__ == "__main__":
    train()
