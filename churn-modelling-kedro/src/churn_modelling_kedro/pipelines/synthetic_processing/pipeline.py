"""
This is a boilerplate pipeline 'synthetic_processing'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import synthetic_data, load_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_data,
            inputs=None,
            outputs="raw_data",
            name="load_csv_file"
        ),
        node(
            func=synthetic_data,
            inputs="raw_data",
            outputs="synthetic_data",
            name="synthetic_data"
        )
    ])
