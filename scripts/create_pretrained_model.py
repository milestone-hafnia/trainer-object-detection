import shutil

from hafnia.dataset.hafnia_dataset_types import TaskInfo
from rfdetr.assets.coco_classes import COCO_CLASSES

from trainer_object_detection.wrapped_model import (
    MODEL_OPTIONS,
    PATH_PRETRAINED_MODELS,
    InitModelConfig,
    primitive_and_model_from_name,
)

if __name__ == "__main__":
    for d in MODEL_OPTIONS:
        if d.pretrained:
            model_weights = "pretrained"
        else:
            model_weights = None

        model_primitive, model_trainer = primitive_and_model_from_name(d.name, model_weights=model_weights)

        # Store model in serializable config format
        max_class_index = max(COCO_CLASSES)
        model_class_names = COCO_CLASSES  # The model.class_names doesn't actually class names
        pretrained_class_names = [f"NotDefined_{i:03d}" for i in range(max_class_index + 1)]
        for class_index, class_name in model_class_names.items():
            pretrained_class_names[class_index] = class_name
        pretrained_model_cfg = InitModelConfig(
            name=d.name,
            task=TaskInfo.from_class_names(primitive=model_primitive, class_names=pretrained_class_names),
            model_weight_path=model_trainer.model_config.pretrain_weights,
        )
        path_model = PATH_PRETRAINED_MODELS / d.name
        shutil.rmtree(path_model, ignore_errors=True)  # Remove old model folder if it exists
        path_model.mkdir(parents=True, exist_ok=True)
        pretrained_model_cfg.save_model(path_model)
