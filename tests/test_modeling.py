import pandas as pd

from src.modeling import (
    build_text_model,
    build_cialdini_model,
)


def create_sample_data():
    return pd.DataFrame({
        "cumulative_history": [
            "urgent account request",
            "urgent payment request",
            "friendly project discussion",
            "friendly team discussion",
            "urgent account review",
            "friendly project update",
        ],
        "authority_present": [1, 1, 0, 0, 1, 0],
        "scarcity_count": [1, 2, 0, 0, 1, 0],
        "label": [1, 1, 0, 0, 1, 0],
    })


def test_text_model_generates_binary_predictions():
    sample_df = create_sample_data()

    model = build_text_model()

    model.fit(
        sample_df["cumulative_history"],
        sample_df["label"],
    )

    predictions = model.predict(
        sample_df["cumulative_history"]
    )

    assert len(predictions) == len(sample_df)
    assert set(predictions).issubset({0, 1})


def test_cialdini_model_generates_binary_predictions():
    sample_df = create_sample_data()

    feature_columns = [
        "authority_present",
        "scarcity_count",
    ]

    model = build_cialdini_model(
        feature_columns
    )

    model.fit(
        sample_df[
            ["cumulative_history", *feature_columns]
        ],
        sample_df["label"],
    )

    predictions = model.predict(
        sample_df[
            ["cumulative_history", *feature_columns]
        ]
    )

    assert len(predictions) == len(sample_df)
    assert set(predictions).issubset({0, 1})