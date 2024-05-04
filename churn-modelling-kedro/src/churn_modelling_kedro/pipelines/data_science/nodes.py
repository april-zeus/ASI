"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""

from pandas import DataFrame, concat
from autogluon.tabular import TabularPredictor
from autogluon.tabular import TabularDataset

def train_model(train_data: TabularDataset) -> TabularPredictor:
    predictor = TabularPredictor(label="Exited", eval_metric="balanced_accuracy").fit(train_data=train_data)
    return predictor

def test_model(predictor: TabularPredictor, test_data: DataFrame) -> DataFrame:
    predictions = DataFrame(predictor.predict(data=test_data, as_pandas=True))
    predictions.rename(columns={"Exited": "Prediction"}, inplace=True)
    predictions = concat([predictions, test_data["Potability"]], axis=1)
    return predictions