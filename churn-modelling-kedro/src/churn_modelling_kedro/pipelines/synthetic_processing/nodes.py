"""
This is a boilerplate pipeline 'synthetic_processing'
generated using Kedro 0.19.5
"""

from pandas import DataFrame, read_csv
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


def load_data() -> DataFrame:
    file_name = "churn-data.csv"
    dataset = read_csv(file_name)
    return dataset


def synthetic_data(train_data: DataFrame, rows: int = 10000) -> DataFrame:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(train_data)

    return synthesizer.sample(num_rows=rows)
