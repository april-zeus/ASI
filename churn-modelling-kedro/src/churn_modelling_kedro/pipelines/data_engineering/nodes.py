"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.5
"""

from pandas import DataFrame, read_csv, qcut, get_dummies
from sklearn.model_selection import train_test_split
from typing import Tuple
from autogluon.tabular import TabularDataset

def load_data() -> DataFrame:
    file_name = "churn-data.csv"
    dataset = read_csv(file_name)
    return dataset

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

    # Robust scaling
    def robust_scaler(variable: DataFrame) -> DataFrame:
        var_median = variable.median()
        quartile1 = variable.quantile(0.25)
        quartile3 = variable.quantile(0.75)
        interquantile_range = quartile3 - quartile1
        if interquantile_range == 0:
            quartile1 = variable.quantile(0.05)
            quartile3 = variable.quantile(0.95)
            interquantile_range = quartile3 - quartile1
            if interquantile_range == 0:
                quartile1 = variable.quantile(0.01)
                quartile3 = variable.quantile(0.99)
                interquantile_range = quartile3 - quartile1
        z = (variable - var_median) / interquantile_range
        return round(z, 3)

    # Identify numerical columns to scale
    columns_to_scale = df.select_dtypes(include=['float64', 'int64']).columns
    like_num = [col for col in columns_to_scale if len(df[col].value_counts()) <= 10]
    columns_to_scale = [col for col in columns_to_scale if col not in ["Gender_Male", "Geography_Germany", "Geography_Spain"]
                        and col != "Exited" and col not in like_num]

    for col in columns_to_scale:
        df[col] = robust_scaler(df[col])

    return df

def split_data(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    train, test = train_test_split(df, test_size=0.1, random_state=42)  # Ensuring reproducibility
    train = TabularDataset(train)
    test = TabularDataset(test)
    return train, test