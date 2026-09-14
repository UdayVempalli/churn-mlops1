import pandas as pd

from src.serve.train import build_pipeline


def test_model_can_train_and_predict():
    X = pd.DataFrame(
        {
            "tenure": [1, 60, 2, 48],
            "MonthlyCharges": [95.0, 25.0, 90.0, 30.0],
            "TotalCharges": [95.0, 1500.0, 180.0, 1400.0],
            "Contract": [
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "One year",
            ],
        }
    )

    y = pd.Series([1, 0, 1, 0])

    model = build_pipeline(X)

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == 4
    assert set(predictions).issubset({0, 1})