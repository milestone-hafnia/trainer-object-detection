from typing import Any, Dict

import mlflow

# The hafnia_benchmark should eventually be added to the hafnia package itself.
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample
from hafnia.dataset.primitives import Bbox
from rfdetr import detr

from trainer_object_detection.hafnia_benchmark import ModelInterface, ModelPrediction
from trainer_object_detection.train_utils import patch_to_support_experiment_tracker_with_hafnia

detr = patch_to_support_experiment_tracker_with_hafnia(detr)


class UserModel(ModelInterface):  # Heavily inspired by cv-benchmark
    def __init__(self, model_config: Dict[str, Any], dataset_info: DatasetInfo):
        self.model = get_model_from_name(model_config.model)
        self.task = dataset_info.get_task_by_primitive(Bbox)

    def load_model_from_config(self, config: Dict[str, Any]):
        pass

    def load_weights(self, path_model: str):
        pass

    def to(self, device: str):  # device: "cpu", "cuda", "mps"
        pass

    # What is the input for predict? Image path, image array, Hafnia Sample, torch.Tensor?
    # Output should always be ModelPrediction
    def predict(self, image) -> ModelPrediction:
        predictions = self.model.predict(image, threshold=0.35)
        bboxes = to_bbox_primitives(predictions, image, self.task.class_names)
        return ModelPrediction(bboxes=bboxes)

    def visualize_gt(self, sample: Sample) -> np.ndarray:
        image = sample.draw_annotations()  # Draw ground truth annotations
        return image

    def visualize_predictions(self, image, prediction: ModelPrediction) -> np.ndarray:
        primitives = []
        for primitive_list in prediction.__dict__.values():
            primitives.extend(primitive_list)
        image = image_visualizations.draw_annotations(image=image, primitives=primitives)
        return image


def get_model_from_name(model_name: str):
    if model_name == "RFDETRBase":
        model = detr.RFDETRBase(pretrain_weights=None)
    elif model_name == "RFDETRNano":
        model = detr.RFDETRNano(pretrain_weights=None)
    elif model_name == "RFDETRSmall":
        model = detr.RFDETRSmall(pretrain_weights=None)
    elif model_name == "RFDETRMedium":
        model = detr.RFDETRMedium(pretrain_weights=None)
    elif model_name == "RFDETRLarge":
        model = detr.RFDETRLarge(pretrain_weights=None)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model


def to_bbox_primitives(predictions, sample: Sample, class_names: list[str]) -> list[Bbox]:
    predictions_bboxes = []
    for bbox, class_idx, confidence in zip(predictions.xyxy, predictions.class_id, predictions.confidence, strict=True):
        bbox = Bbox(
            height=(bbox[3] - bbox[1]) / sample.height,
            width=(bbox[2] - bbox[0]) / sample.width,
            top_left_x=bbox[0] / sample.width,
            top_left_y=bbox[1] / sample.height,
            class_idx=int(class_idx),
            class_name=class_names[int(class_idx)],
            confidence=float(confidence),
            ground_truth=False,
        )
        predictions_bboxes.append(bbox)
    return predictions_bboxes


def safe_index(arr, idx):
    return arr[idx] if 0 <= idx < len(arr) else None


class MetricsTensorBoardSinkMLflow:
    """
    Replacement for MetricsTensorBoardSink that logs to MLflow instead of TensorBoard.
    Keeps the same interface: __init__, update, close.
    """

    def __init__(self, output_dir: str):
        print("MLflow Metrics sink initialized")

    def update(self, values: dict):
        epoch = values["epoch"]

        # losses
        if "train_loss" in values:
            mlflow.log_metric("Loss/Train", values["train_loss"], step=epoch)
        if "test_loss" in values:
            mlflow.log_metric("Loss/Test", values["test_loss"], step=epoch)

        # standard COCO eval
        if "test_coco_eval_bbox" in values:
            coco_eval = values["test_coco_eval_bbox"]
            ap50_90 = safe_index(coco_eval, 0)
            ap50 = safe_index(coco_eval, 1)
            ar50_90 = safe_index(coco_eval, 8)
            if ap50_90 is not None:
                mlflow.log_metric("Metrics/Base/AP50_90", ap50_90, step=epoch)
            if ap50 is not None:
                mlflow.log_metric("Metrics/Base/AP50", ap50, step=epoch)
            if ar50_90 is not None:
                mlflow.log_metric("Metrics/Base/AR50_90", ar50_90, step=epoch)

        # EMA COCO eval
        if "ema_test_coco_eval_bbox" in values:
            ema_coco_eval = values["ema_test_coco_eval_bbox"]
            ema_ap50_90 = safe_index(ema_coco_eval, 0)
            ema_ap50 = safe_index(ema_coco_eval, 1)
            ema_ar50_90 = safe_index(ema_coco_eval, 8)
            if ema_ap50_90 is not None:
                mlflow.log_metric("Metrics/EMA/AP50_90", ema_ap50_90, step=epoch)
            if ema_ap50 is not None:
                mlflow.log_metric("Metrics/EMA/AP50", ema_ap50, step=epoch)
            if ema_ar50_90 is not None:
                mlflow.log_metric("Metrics/EMA/AR50_90", ema_ar50_90, step=epoch)

    def save(self):
        pass

    def close(self):
        pass


def patch_to_support_experiment_tracker_with_hafnia(detr: types.ModuleType):
    detr.MetricsPlotSink = MetricsTensorBoardSinkMLflow

    return detr
