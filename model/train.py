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

# Synthetic spam dataset for reproducible CI runs.
# Balanced 50 spam / 50 ham (100 total) to give TF-IDF+LR enough
# training signal with a standard 80/20 split (80 train, 20 test).
SYNTHETIC_DATA = [
    # ── SPAM (label=1) ────────────────────────────────────────────────────
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
    ("Exclusive deal! Earn $500 daily from home, guaranteed!", 1),
    ("Click here to claim your free vacation package!", 1),
    ("You have won a lottery! Provide details to claim prize.", 1),
    ("Cheap Viagra! No prescription required. Order now.", 1),
    ("Act now! Limited seats for our wealth seminar. FREE entry.", 1),
    ("Double your investment in 24 hours! Guaranteed returns.", 1),
    ("Congratulations, your PayPal account needs verification!", 1),
    ("FREE gift cards! Take our 30-second survey to qualify.", 1),
    ("Your bank account has suspicious activity. Click to verify.", 1),
    ("Earn passive income from crypto. Join now, it's FREE!", 1),
    ("WINNER! You've been randomly selected for a $5000 prize.", 1),
    ("Hot singles want to meet you! Sign up free today.", 1),
    ("Lose 30 pounds in 30 days with this miracle supplement!", 1),
    ("Your Netflix subscription has expired. Update billing now.", 1),
    ("We detected a virus on your PC. Call this number NOW.", 1),
    ("Claim your inheritance of $4.5 million. Reply for details.", 1),
    ("100% FREE online casino! Win real money tonight.", 1),
    ("Your Amazon account is locked. Click here immediately.", 1),
    ("Work from home and earn $3000/week. No experience needed.", 1),
    ("URGENT: IRS notice. You owe back taxes. Call now to avoid arrest.", 1),
    ("Buy 1 get 10 free! Limited stock. Order before midnight.", 1),
    ("You qualify for a government grant. No repayment needed!", 1),
    ("Enlarge and impress! Discreet shipping worldwide.", 1),
    ("Your package is held. Pay $1.99 customs fee to release it.", 1),
    ("Congratulations! You're our 1 millionth visitor. Claim now!", 1),
    ("This investment tip made me $10k last week. See how.", 1),
    ("FREE iPhone 15! Complete this offer to claim yours.", 1),
    ("Your credit score can be boosted overnight. Learn how!", 1),
    ("Debt relief program: Erase $20k of debt legally!", 1),
    ("SPECIAL OFFER: Rolex watches from $49. Authentic guaranteed.", 1),
    ("Final notice: Your social security number has been suspended.", 1),
    ("Diet secret celebrities don't want you to know about!", 1),
    ("Earn Bitcoin daily! Automated trading bot, 100% profit.", 1),
    ("Click NOW to unsubscribe or you will be charged $99/month.", 1),
    ("You've been pre-approved for a $50,000 personal loan!", 1),
    ("Make thousands per week selling on Amazon. FREE webinar!", 1),
    ("ALERT: Your email will be deactivated. Verify now!", 1),
    ("Win big with our online slots. No deposit bonus inside!", 1),
    ("Your Microsoft account has been compromised. Act fast!", 1),
    ("Get a university degree online in 2 weeks! Accredited.", 1),
    # ── HAM (label=0) ─────────────────────────────────────────────────────
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
    ("Attached is the invoice for last month's services.", 0),
    ("Could you send me the updated design files when ready?", 0),
    ("The CI pipeline passed. Ready to merge when you approve.", 0),
    ("Great work on the presentation today, really impressed.", 0),
    ("I've pushed the hotfix to the staging branch for review.", 0),
    ("See you at the 10am planning session tomorrow.", 0),
    ("The client approved the proposal. We're good to proceed.", 0),
    ("Do you have the login credentials for the dev database?", 0),
    ("Lunch is on me today — great job closing that deal.", 0),
    ("Your flight confirmation for March 12 is attached.", 0),
    ("The quarterly results have been posted to the intranet.", 0),
    ("We're moving the office to the 4th floor next Monday.", 0),
    ("I'll send over the contract for signature this afternoon.", 0),
    ("Quick question: which library are we using for auth?", 0),
    ("Your annual performance review is scheduled for Friday.", 0),
    ("The server maintenance window is tonight from 2-4am.", 0),
    ("Coffee catch-up this week? I have a few ideas to share.", 0),
    ("Here are the action items from yesterday's meeting.", 0),
    ("The new employee handbook has been updated on Confluence.", 0),
    ("Your expense report has been approved and will be reimbursed.", 0),
    ("Joining in 5 — stuck in traffic, please start without me.", 0),
    ("The sprint board has been updated with today's tickets.", 0),
    ("Could you proofread this draft before I send it out?", 0),
    ("We'll need to extend the deadline by two days — thoughts?", 0),
    ("The team lunch is booked for Friday at noon.", 0),
    ("Your AWS bill for February is ready to view.", 0),
    ("Just merged the feature branch. Tests are all green.", 0),
    ("Would you be available for a quick call around 2pm?", 0),
    ("The new API documentation is live on the developer portal.", 0),
    ("Reminder: submit your timesheets before end of day Friday.", 0),
    ("We hit 10k users this week — great milestone for the team!", 0),
    ("I've updated the Jira tickets with the latest estimates.", 0),
    ("Your background check has been completed successfully.", 0),
    ("Let's sync up briefly before the client call at 3pm.", 0),
    ("The product roadmap has been shared with all stakeholders.", 0),
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
