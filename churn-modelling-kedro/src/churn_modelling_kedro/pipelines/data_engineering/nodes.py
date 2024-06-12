"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.5
"""
import pandas as pd
from pandas import DataFrame, get_dummies, concat
from sklearn.preprocessing import LabelEncoder


def process_data(df: DataFrame) -> DataFrame:
    # Handle missing values
    df = df.dropna()  # drop rows with missing values

    # Feature engineering
    df["NewTenure"] = df["Tenure"] / df["Age"]

    print(df)

    lab_enc_2 = LabelEncoder()
    df["Gender"] = lab_enc_2.fit_transform(df["Gender"])

    lab_enc = LabelEncoder()
    df["Geography"] = lab_enc.fit_transform(df["Geography"])

    # Dropping unnecessary variables
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    print(df)

    return df


