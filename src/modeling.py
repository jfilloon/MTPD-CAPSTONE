"""Reusable model-pipeline construction functions."""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_text_model():
    """Build a TF-IDF logistic-regression text model."""

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def build_cialdini_model(
    cialdini_feature_columns,
    text_column="cumulative_history",
):
    """Build a history-text and Cialdini-feature model."""

    feature_processor = ColumnTransformer([
        (
            "history_text",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
            text_column,
        ),
        (
            "cialdini_features",
            "passthrough",
            list(cialdini_feature_columns),
        ),
    ])

    return Pipeline([
        ("features", feature_processor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])