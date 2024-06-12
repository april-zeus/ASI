"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""
import pickle
from pandas import concat
from autogluon.tabular import TabularPredictor
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Tuple
from autogluon.tabular import TabularDataset


def split_data(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    train, test = train_test_split(df, test_size=0.1, random_state=42)  # Ensuring reproducibility
    train = TabularDataset(train)
    test = TabularDataset(test)
    return train, test


def train_model(synthetic_train_data: TabularDataset) -> TabularPredictor:
    predictor = TabularPredictor(label="Exited", eval_metric="balanced_accuracy").fit(train_data=synthetic_train_data,
                                                                                      keep_only_best=True)
    return predictor


def test_model(predictor: TabularPredictor, test_data: DataFrame) -> DataFrame:
    predictions = DataFrame(predictor.predict(data=test_data, as_pandas=True))
    predictions.rename(columns={"Exited": "Prediction"}, inplace=True)
    predictions = concat([predictions, test_data["Exited"]], axis=1)
    return predictions


def save_model(predictor: TabularPredictor):
    filename = "predictor.pkl"
    predictor.persist()
    pickle.dump(predictor, open(filename, "wb"))
