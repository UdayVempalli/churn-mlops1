import pandas as pd

from src.serve.prep import clean_data


def test_clean_data_removes_customer_id():
    df = pd.DataFrame(
        {
            "customerID": ["ABC-123"],
            "tenure": [1],
            "MonthlyCharges": [50.0],
            "TotalCharges": ["50.0"],
            "Churn": ["Yes"],
        }
    )

    result = clean_data(df)

    assert "customerID" not in result.columns
def test_clean_data_maps_churn_to_numbers():
    df = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "tenure": [1, 10],
            "MonthlyCharges": [50.0, 60.0],
            "TotalCharges": ["50.0", "600.0"],
            "Churn": ["Yes", "No"],
        }
    )

    result = clean_data(df)

    assert result["Churn"].tolist() == [1, 0]
def test_clean_data_fills_blank_total_charges_with_zero():
    df = pd.DataFrame(
        {
            "customerID": ["A"],
            "tenure": [0],
            "MonthlyCharges": [70.0],
            "TotalCharges": [" "],
            "Churn": ["No"],
        }
    )

    result = clean_data(df)

    assert result["TotalCharges"].iloc[0] == 0.0