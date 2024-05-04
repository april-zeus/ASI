"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.5
"""

from pandas import read_csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Tuple
from autogluon.tabular import TabularDataset

def load_data():
    file_name = "churn-data.csv"
    dataset = read_csv(file_name)
    return dataset

# def process_data(data: DataFrame) -> Tuple[DataFrame, DataFrame]):
#     TODO: przygotować dane do regresji jak na SUMLach

def split_data(data: DataFrame) -> Tuple[DataFrame, DataFrame]:
    train, test = train_test_split(data, test_size=0.1)  # Assuming a 90-10 split
    train = TabularDataset(train)
    test = TabularDataset(test)
    return train, test