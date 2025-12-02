from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# The hafnia_benchmark should eventually be added to the hafnia package itself.
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample
from hafnia.dataset.primitives import Bbox, Bitmask, Classification, Polygon
from hafnia.experiment import HafniaLogger
from hafnia.utils import progress_bar
from hafnia.visualizations import image_visualizations
from pydantic import BaseModel, Field


class ModelPrediction(BaseModel):
    classifications: Optional[List[Classification]] = Field(
        default=None, description="Optional list of classifications"
    )
    bboxes: Optional[List[Bbox]] = Field(default=None, description="Optional list of bounding boxes")
    bitmasks: Optional[List[Bitmask]] = Field(default=None, description="Optional list of bitmasks")
    polygons: Optional[List[Polygon]] = Field(default=None, description="Optional list of polygons")


class ModelInterface:  # Heavily inspired by cv-benchmark
    def __init__(self, model_config: Dict[str, Any], dataset_info: DatasetInfo):
        pass

    def load_weights(self, path_model: str):
        pass

    def to(self, device: str):  # device: "cpu", "cuda", "mps"
        pass

    # What is the input for predict? Image path, image array, Hafnia Sample, torch.Tensor?
    # Output should always be ModelPrediction
    def predict(self, image) -> ModelPrediction:
        pass

    def visualize_gt(self, sample: Sample) -> np.ndarray:
        image = sample.draw_annotations()  # Draw ground truth annotations
        return image

    def visualize_predictions(self, image, prediction: ModelPrediction) -> np.ndarray:
        primitives = []
        for primitive_list in prediction.__dict__.values():
            primitives.extend(primitive_list)
        image = image_visualizations.draw_annotations(image=image, primitives=primitives)
        return image


def calculate_benchmark_metrics(
    list_sample_and_predictions: List[Tuple[Sample, ModelPrediction]],
    dataset_info: DatasetInfo,
) -> Dict[str, float]:
    # Placeholder for actual metric calculations.
    metrics = {}
    bbox_tasks = dataset_info.get_tasks_by_primitive(Bbox)
    for task in bbox_tasks:
        # Use either task_name or task_primitive
        metric_prefix = task.primitive.__name__
        metrics.update(
            {
                f"{metric_prefix}/mAP": 0.75,  # Example value
                f"{metric_prefix}/precision": 0.80,  # Example value
                f"{metric_prefix}/recall": 0.70,  # Example value
            }
        )

    bitmask_tasks = dataset_info.get_tasks_by_primitive(Bitmask)
    for task in bitmask_tasks:
        metric_prefix = task.primitive.__name__
        metrics.update(
            {
                f"{metric_prefix}/mAP": 0.65,  # Example value
                f"{metric_prefix}/precision": 0.70,  # Example value
                f"{metric_prefix}/recall": 0.60,  # Example value
            }
        )

    polygon_tasks = dataset_info.get_tasks_by_primitive(Polygon)
    for task in polygon_tasks:
        metric_prefix = task.primitive.__name__
        metrics.update(
            {
                f"{metric_prefix}/mAP": 0.68,  # Example value
                f"{metric_prefix}/precision": 0.72,  # Example value
                f"{metric_prefix}/recall": 0.62,  # Example value
            }
        )

    classification_tasks = dataset_info.get_tasks_by_primitive(Classification)
    for task in classification_tasks:
        metric_prefix = task.primitive.__name__
        metrics.update(
            {
                f"{metric_prefix}/accuracy": 0.85,  # Example value
                f"{metric_prefix}/precision": 0.88,  # Example value
                f"{metric_prefix}/recall": 0.80,  # Example value
            }
        )
    return metrics


def benchmark_model(
    benchmark_dataset: HafniaDataset,
    model_wrapper: ModelInterface,
    hyperparams: Optional[Dict[str, Any]],  # Hyperparameters to log in the benchmark
):
    logger = HafniaLogger(project_name="Benchmark", log_dir=".data/benchmarks/")
    if hyperparams is not None:
        logger.log_configuration(hyperparams)
    t_diffs = []
    list_sample_and_predictions: List[Tuple[Sample, ModelPrediction]] = []
    for dict_sample in progress_bar(benchmark_dataset):
        sample = Sample(**dict_sample)
        image = sample.read_image()

        # Run and time prediction
        t0 = time.perf_counter()
        prediction = model_wrapper.predict(image)
        t_diff = time.perf_counter() - t0

        t_diffs.append(t_diff)
        list_sample_and_predictions.append((sample, prediction))

    metrics = calculate_benchmark_metrics(list_sample_and_predictions, benchmark_dataset.info)
    logger.log_metric(metrics)
