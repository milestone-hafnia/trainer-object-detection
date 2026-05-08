import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, Union

import torch
from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.hafnia_dataset_types import Bitmask, ModelInfo, TaskInfo
from hafnia.dataset.primitives import Bbox, Primitive
from hafnia.log import user_logger
from pydantic import BaseModel
from rfdetr import config, detr
from rfdetr.assets.model_weights import download_pretrain_weights

MODEL_CONFIG_NAME = "model_config.json"


@dataclass
class ModelOption:
    name: str
    pretrained: bool
    supported: bool


MODEL_OPTIONS = [
    ModelOption(name="RFDETRNano", pretrained=True, supported=True),
    # ModelOption(name="RFDETRSmall", pretrained=True, supported=True),
    ModelOption(name="RFDETRMedium", pretrained=True, supported=True),
    ModelOption(name="RFDETRLarge", pretrained=True, supported=True),
    ModelOption(name="RFDETRSegNano", pretrained=True, supported=True),
]
PATH_PRETRAINED_MODELS = Path(__file__).parent.parent.parent / "pretrained_models"


class InitModelConfig(BaseModel):
    name: str
    task: TaskInfo
    model_weight_path: Optional[str]

    def get_trainer(self):
        _, model_trainer = primitive_and_model_from_name(self.name, model_weights=self.model_weight_path)
        return model_trainer

    def save_model(self, path_model_folder: Path):
        path_model_folder.mkdir(parents=True, exist_ok=True)
        # Path should be relative
        if self.model_weight_path is not None:
            model_path = Path(self.model_weight_path).name
            new_model_weight_path = path_model_folder / model_path
            shutil.copy2(self.model_weight_path, new_model_weight_path)
            self.model_weight_path = model_path  # Relative path in the config json

        path_json_file = path_model_folder / MODEL_CONFIG_NAME
        config_json = self.model_dump_json(indent=4)
        path_json_file.write_text(config_json)

    @staticmethod
    def load_model(path_json: Union[str, Path], use_weights: bool) -> "InitModelConfig":
        path_json = Path(path_json)
        model_config_dict = json.loads(path_json.read_text())
        model_config = InitModelConfig.model_validate(model_config_dict)
        if model_config.model_weight_path is not None:
            model_config.model_weight_path = (path_json.parent / model_config.model_weight_path).as_posix()

        if use_weights and model_config.model_weight_path is None:
            user_logger.warning(
                f"The specified model '{path_json}' does not have pretrained weights available, but "
                "'pretrained=True' was set. The model will be trained from scratch."
            )

        if not use_weights and model_config.model_weight_path is not None:
            user_logger.warning(
                f"The specified model '{path_json}' has pretrained weights available, but "
                "'pretrained=False' was set. The model will be trained from scratch without using the pretrained weights."
            )
        return model_config


class InferenceConfig(BaseModel):
    compile: bool = True
    batch_size: int = 1
    threshold: float = 0.05


class WrappedModel(InferenceModel):
    def __init__(self, model: detr.RFDETR, task: TaskInfo, inference_config: InferenceConfig):
        self.model = model
        self.task = task
        self.inference_config = inference_config

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name=self.model.__class__.__name__, tasks=[self.task])

    def optimize_for_inference(self):
        self.model.optimize_for_inference(
            compile=self.inference_config.compile,
            batch_size=self.inference_config.batch_size,
            dtype=torch.float32,
        )

    def predict(self, images: Union[ImageType, List[ImageType]], sample_dict: Optional[dict] = None) -> List[Primitive]:
        predictions = self.model.predict(images, threshold=self.inference_config.threshold)
        bboxes: List[Bbox] = to_bbox_primitives(predictions, images.shape[:2], bbox_task=self.task)
        return bboxes

    @staticmethod
    def load_model(path_json: Union[str, Path], inference_config: InferenceConfig) -> "WrappedModel":
        path_json = Path(path_json)
        model_config_dict = json.loads(path_json.read_text())
        model_config = InitModelConfig.model_validate(model_config_dict)

        model_weight_path = None
        if model_config.model_weight_path is not None:
            model_weight_path = path_json.parent / model_config.model_weight_path

        primitive, model = primitive_and_model_from_name(model_config.name, model_weights=str(model_weight_path))

        if primitive != model_config.task.primitive:
            raise ValueError(
                f"Model '{model_config.name}' is associated with primitive '{primitive.__name__}', "
                f"but the task in the config file requires primitive '{model_config.task.primitive.__name__}'."
            )

        return WrappedModel(model=model, task=model_config.task, inference_config=inference_config)


def primitive_and_model_from_name(
    model_name: str, model_weights: Optional[str] = "pretrained"
) -> Tuple[
    Type[Primitive],
    detr.RFDETR,
]:

    if model_name == "RFDETRNano":
        primitive = Bbox
        model_class = detr.RFDETRNano
        model_config: config.RFDETRBaseConfig = config.RFDETRNanoConfig()

    elif model_name == "RFDETRSmall":
        primitive = Bbox
        model_class = detr.RFDETRSmall
        model_config: config.RFDETRBaseConfig = config.RFDETRSmallConfig()

    elif model_name == "RFDETRMedium":
        primitive = Bbox
        model_class = detr.RFDETRMedium
        model_config: config.RFDETRBaseConfig = config.RFDETRMediumConfig()

    elif model_name == "RFDETRLarge":
        primitive = Bbox
        model_class = detr.RFDETRLarge
        model_config: config.RFDETRBaseConfig = config.RFDETRLargeConfig()

    elif model_name == "RFDETRSegNano":
        primitive = Bitmask
        model_class = detr.RFDETRSegNano
        model_config: config.RFDETRBaseConfig = config.RFDETRSegNanoConfig()
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    kwargs: dict[str, Any] = {}

    if model_weights == "pretrained":
        if not Path(model_config.pretrain_weights).exists():
            download_pretrain_weights(model_config.pretrain_weights)
        kwargs["pretrain_weights"] = model_config.pretrain_weights
    else:
        kwargs["pretrain_weights"] = model_weights
    model = model_class(**kwargs)
    return primitive, model


def to_bbox_primitives(predictions, image_shape: Tuple[int, int], bbox_task: TaskInfo) -> list[Bbox]:
    predictions_bboxes = []
    for bbox, class_idx, confidence in zip(predictions.xyxy, predictions.class_id, predictions.confidence, strict=True):
        # Model creates n+1 class indices, where the last index is "no object" or "__background__" class
        is_background_class = class_idx.item() == len(bbox_task.classes)
        if is_background_class:
            continue
        bbox = Bbox(
            height=(bbox[3] - bbox[1]) / image_shape[0],
            width=(bbox[2] - bbox[0]) / image_shape[1],
            top_left_x=bbox[0] / image_shape[1],
            top_left_y=bbox[1] / image_shape[0],
            class_idx=int(class_idx),
            class_name=bbox_task.classes[int(class_idx)].name,
            confidence=float(confidence),
            ground_truth=False,
        )
        predictions_bboxes.append(bbox)
    return predictions_bboxes
