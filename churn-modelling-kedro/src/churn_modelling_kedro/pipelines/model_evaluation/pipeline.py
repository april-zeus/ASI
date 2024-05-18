from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model,
            inputs="predictions",
            outputs=["evaluation_metrics", "confusion_matrix"],
            name="autogluon_evaluation"
            )
    ])