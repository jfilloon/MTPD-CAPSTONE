from src.evaluation import evaluate_predictions


def test_evaluate_predictions():
    results = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        model_name="Test model",
    )

    assert results["model"] == "Test model"
    assert results["accuracy"] == 0.5
    assert results["precision"] == 0.5
    assert results["recall"] == 0.5
    assert results["f1_score"] == 0.5
    assert results["false_positive_rate"] == 0.5
    assert results["false_negative_rate"] == 0.5
    assert results["true_negatives"] == 1
    assert results["false_positives"] == 1
    assert results["false_negatives"] == 1
    assert results["true_positives"] == 1
    