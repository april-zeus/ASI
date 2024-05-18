from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model,
            inputs=["predictions", "test_labels"],
            outputs="evaluation_plot",
            name="node_model_evaluation"
            )
    ])