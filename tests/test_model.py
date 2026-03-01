"""
Tests for the spam classifier pipeline.

Validates:
- Model trains successfully and returns a pipeline object
- Accuracy meets minimum threshold
- Predictions are within valid output range {0, 1}
- Model handles edge-case inputs without crashing
- predict_proba outputs are valid probabilities
"""

import sys
import os
import pickle
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.train import load_data, build_pipeline, train, MODEL_PATH


@pytest.fixture(scope="module")
def trained_model_and_metrics():
    """Train once for all tests in this module."""
    model, metrics = train()
    return model, metrics


class TestDataLoading:
    def test_load_data_returns_dataframe(self):
        df = load_data()
        assert df is not None
        assert len(df) > 0

    def test_data_has_expected_columns(self):
        df = load_data()
        assert "text" in df.columns
        assert "label" in df.columns

    def test_labels_are_binary(self):
        df = load_data()
        assert set(df["label"].unique()).issubset({0, 1})

    def test_data_has_both_classes(self):
        df = load_data()
        assert df["label"].nunique() == 2

    def test_no_empty_texts(self):
        df = load_data()
        assert df["text"].str.strip().ne("").all()


class TestModelTraining:
    def test_train_returns_model_and_metrics(self, trained_model_and_metrics):
        model, metrics = trained_model_and_metrics
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metrics_keys_present(self, trained_model_and_metrics):
        _, metrics = trained_model_and_metrics
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics

    def test_accuracy_above_threshold(self, trained_model_and_metrics):
        """Model must achieve at least 70% accuracy on the held-out split."""
        _, metrics = trained_model_and_metrics
        assert metrics["accuracy"] >= 0.70, (
            f"Accuracy {metrics['accuracy']:.2f} is below minimum threshold of 0.70"
        )

    def test_f1_above_threshold(self, trained_model_and_metrics):
        """F1 score must be at least 0.60 to ensure recall/precision balance."""
        _, metrics = trained_model_and_metrics
        assert metrics["f1"] >= 0.60, (
            f"F1 {metrics['f1']:.2f} is below minimum threshold of 0.60"
        )

    def test_metrics_in_valid_range(self, trained_model_and_metrics):
        _, metrics = trained_model_and_metrics
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"Metric '{key}' = {val} is out of [0, 1] range"

    def test_model_artifact_saved(self, trained_model_and_metrics):
        assert os.path.exists(MODEL_PATH), f"Model artifact not found at {MODEL_PATH}"


class TestPredictions:
    def test_predictions_are_binary(self, trained_model_and_metrics):
        model, _ = trained_model_and_metrics
        samples = [
            "Win a free prize now!!!",
            "Hey, let's catch up tomorrow.",
        ]
        preds = model.predict(samples)
        assert set(preds).issubset({0, 1}), f"Predictions contain non-binary values: {preds}"

    def test_spam_sample_predicted_as_spam(self, trained_model_and_metrics):
        model, _ = trained_model_and_metrics
        spam = ["FREE MONEY CLICK NOW WIN PRIZE URGENT!!!"]
        pred = model.predict(spam)[0]
        assert pred == 1, "Clear spam sample was not classified as spam"

    def test_ham_sample_predicted_as_ham(self, trained_model_and_metrics):
        model, _ = trained_model_and_metrics
        ham = ["Please send me the meeting agenda for tomorrow."]
        pred = model.predict(ham)[0]
        assert pred == 0, "Clear ham sample was not classified as ham"

    def test_output_shape_matches_input(self, trained_model_and_metrics):
        model, _ = trained_model_and_metrics
        inputs = ["message one", "message two", "message three"]
        preds = model.predict(inputs)
        assert len(preds) == len(inputs)

    def test_predict_proba_valid_probabilities(self, trained_model_and_metrics):
        model, _ = trained_model_and_metrics
        samples = ["win free money", "meeting at 3pm"]
        proba = model.predict_proba(samples)
        assert proba.shape == (2, 2), f"Expected shape (2, 2), got {proba.shape}"
        assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities do not sum to 1"
        assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities out of [0, 1]"

    def test_empty_string_does_not_crash(self, trained_model_and_metrics):
        model, _ = trained_model_and_metrics
        try:
            preds = model.predict([""])
            assert len(preds) == 1
        except Exception as e:
            pytest.fail(f"Model crashed on empty string input: {e}")


class TestSavedArtifact:
    def test_saved_model_loads_and_predicts(self, trained_model_and_metrics):
        """Verify the pickled artifact loads correctly and produces valid predictions."""
        _ = trained_model_and_metrics  # ensure artifact is written first
        with open(MODEL_PATH, "rb") as f:
            loaded_model = pickle.load(f)
        preds = loaded_model.predict(["win a prize now", "see you at lunch"])
        assert len(preds) == 2
        assert set(preds).issubset({0, 1})
