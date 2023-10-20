
import pandas as pd


def test_data():

    # Make sure that when you load the manual, there are no NaN values
    df = pd.read_csv("./inputs/excon_processed_manual.csv", sep='|', encoding="utf-8", na_filter=False)
    assert not df.isna().any().any()

