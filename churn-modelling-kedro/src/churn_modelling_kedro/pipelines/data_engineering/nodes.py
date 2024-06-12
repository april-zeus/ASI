"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.5
"""
from pandas import DataFrame, read_csv, qcut, get_dummies

def process_data(df: DataFrame) -> DataFrame:
    # Handle missing values
    df = df.dropna()  # drop rows with missing values

    # Feature engineering
    df["NewTenure"] = df["Tenure"] / df["Age"]
    df["NewCreditsScore"] = qcut(df['CreditScore'], 6, labels=[1, 2, 3, 4, 5, 6])
    df["NewAgeScore"] = qcut(df['Age'], 8, labels=[1, 2, 3, 4, 5, 6, 7, 8])
    df["NewBalanceScore"] = qcut(df['Balance'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    df["NewEstSalaryScore"] = qcut(df['EstimatedSalary'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # One hot encoding
    columns_to_encode = ["Gender", "Geography"]
    df = get_dummies(df, columns=columns_to_encode, drop_first=True)

    # Dropping unnecessary variables
    df = df.drop(["CustomerId", "Surname"], axis=1)

    return df


