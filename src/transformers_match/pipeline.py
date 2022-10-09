"""
This is a boilerplate pipeline
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_load, _pseudo_train, _progress_worker


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=model_load,
                inputs="params:model_default",
                outputs="model",
                name="load",
            ),
            node(
                func=_pseudo_train,
                inputs=["model", "params:model_default"],
                outputs="leveraged_model",
                name="train",
            ),
            node(
                func=_progress_worker,
                inputs=["leveraged_model", "queries", "corpus", "params:model_default"],
                outputs="matched_dataset",
                name="sentence_transforming",
            ),
        ]
    )
