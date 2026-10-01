from model import train_model

def test_model_accuracy():
    _, accuracy, _, _, _ = train_model()
    assert accuracy >= 0.90

def test_prediction_count():
    _, _, X_test, _, predictions = train_model()
    assert len(predictions) == len(X_test)

def test_prediction_labels():
    _, _, _, _, predictions = train_model()
    assert all(prediction in [0, 1, 2] for prediction in predictions)
