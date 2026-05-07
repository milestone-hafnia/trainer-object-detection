import shutil
from pathlib import Path
from typing import Annotated, Optional, Type

import polars as pl
import torch
from cyclopts import App, Parameter
from hafnia import utils as hafnia_utils
from hafnia.dataset.benchmark.benchmark import metric_calculations, run_inference_on_dataset
from hafnia.dataset.hafnia_dataset import HafniaDataset, SplitName, TaskInfo
from hafnia.dataset.primitives import Primitive
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from rfdetr import detr

import trainer_object_detection.wrapped_model
from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import MODEL_CONFIG_NAME, InferenceConfig, InitModelConfig, WrappedModel

detr = utils.patch_to_support_experiment_tracker_with_hafnia(detr)
default_inference_config = InferenceConfig(compile=True, batch_size=1, threshold=0.05)

app = App(name="train", help="PyTorch Training")

MODEL_NAME_OPTIONS = [f"pretrained_models/{d.name}" for d in trainer_object_detection.wrapped_model.MODEL_OPTIONS]

DEFAULT_INFERENCE_MODEL = "checkpoint_best_ema"
INFERENCE_MODEL_OPTIONS = [DEFAULT_INFERENCE_MODEL, "checkpoint_best_regular", "checkpoint_best_total"]


@app.default
def main(
    project_name: Annotated[str, Parameter(help="Project name for the experiment")] = "Trainer RF-DETR",
    model_path: Annotated[
        str, Parameter(help=f"Model name or checkpoint path. Options: {MODEL_NAME_OPTIONS}")
    ] = "./pretrained_models/RFDETRNano",
    pretrained: Annotated[bool, Parameter(help="Use pretrained weights")] = True,
    epochs: Annotated[int, Parameter(help="Number of epochs to train")] = 10,
    batch_size: Annotated[int, Parameter(help="Batch size for training")] = 8,
    grad_accumulation_steps: Annotated[int, Parameter(help="Gradient accumulation steps")] = 1,
    learning_rate: Annotated[float, Parameter(help="Learning rate for optimizer")] = 0.001,
    resolution: Annotated[
        Optional[int],
        Parameter(help="Input resolution (square side in pixels). Defaults to each model's built-in value."),
    ] = None,
    task_name: Annotated[
        Optional[str],
        Parameter(
            help="Dataset task name used for training the model. Use only this if the dataset has multiple tasks"
        ),
    ] = None,
    samples: Annotated[
        Optional[int], Parameter(help="Number of samples to use for training. Use for testing purposes.")
    ] = -1,
    stop_early: Annotated[
        bool,
        Parameter(help="Break script before training starts. Can be used to avoid long training times during testing."),
    ] = False,
    inference_model_name: Annotated[
        str,
        Parameter(help=f"Inference model name or checkpoint path. Options: {INFERENCE_MODEL_OPTIONS}"),
    ] = DEFAULT_INFERENCE_MODEL,
    inference_config: Annotated[
        InferenceConfig, Parameter(help="Inference configuration for the model")
    ] = default_inference_config,
):
    # Check cuda availability
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    logger = HafniaLogger(project_name=project_name)

    if hafnia_utils.is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = hafnia_utils.get_dataset_path_in_hafnia_cloud()  # The path to hidden dataset is returned
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        dataset = HafniaDataset.from_name("midwest-detection-traffic", version="1.0.0")

    if samples is not None and samples > 0:
        dataset = dataset.select_samples(n_samples=samples)

    inference_model_json = Path(model_path) / MODEL_CONFIG_NAME
    model_config = InitModelConfig.load_model(inference_model_json, use_weights=pretrained)
    model_primitive = model_config.task.primitive

    model_trainer = model_config.get_trainer()
    configuration = {
        "model": model_path,
        "pretrained": pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "dataset": dataset.info.dataset_name,
        "has_cuda": has_cuda,
    }
    logger.log_configuration(configuration)

    task_info = get_dataset_task_from_model_primitive(dataset, model_primitive, task_name)

    dataset_test = dataset.create_split_dataset(split_name=SplitName.TEST)
    dataset_train_val = dataset.create_split_dataset(split_name=[SplitName.TRAIN, SplitName.VAL])
    dataset_train_val = remove_images_with_no_bboxes(dataset_train_val, model_primitive=model_primitive)

    # Convert dataset to COCO format for training
    dataset_name = dataset_train_val.info.dataset_name
    dataset_path = Path(".data") / f"format_coco_roboflow_{dataset_name}"
    dataset_train_val.to_coco_format(dataset_path, task_name=task_info.name)
    path_experiment = logger._local_experiment_path
    path_experiment.mkdir(parents=True, exist_ok=True)

    if stop_early:
        user_logger.info("Early stopping before training was activated with '--stop_early' flag.")
        return None

    model_trainer.train(
        dataset_dir=dataset_path.as_posix(),
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        grad_accum_steps=grad_accumulation_steps,
        output_dir=path_experiment.as_posix(),
        resolution=resolution,
    )

    model_folder_path = logger.path_model()
    # Move final model weights to model folder (e.g. "checkpoint_best_regular.pth" and "checkpoint_best_total.pth")
    final_models = list(path_experiment.glob("checkpoint_*.pth"))
    model_path = {}
    for checkpoint_path in final_models:
        model_checkpoint_path = model_folder_path / checkpoint_path.name
        model_config = InitModelConfig(name=model_config.name, task=task_info, model_weight_path=str(checkpoint_path))
        model_config.save_model(model_checkpoint_path)
        model_path[checkpoint_path.stem] = model_checkpoint_path

    checkpoint_model_paths = final_models  # For now we simply add final models as checkpoints
    checkpoints_folder_path = logger.path_model_checkpoints()
    for ckpt_path in checkpoint_model_paths:
        model_config = InitModelConfig(name=model_config.name, task=task_info, model_weight_path=str(ckpt_path))
        model_config.save_model(checkpoints_folder_path / ckpt_path.stem)

    # Move files to artifact folder
    artifact_folder_path = logger._path_artifacts()
    check_for_files = ["log.txt", "metrics.csv", "results.json"]
    for file_pattern in check_for_files:
        file_paths = list(path_experiment.glob(file_pattern))
        if len(file_paths) == 0:
            user_logger.warning(f"No files found for pattern: {file_pattern}")
        for file_path in file_paths:
            shutil.copy2(file_path, artifact_folder_path)

    #### 'TEST' split inference/benchmarking ####
    inference_model_json = model_path[inference_model_name] / MODEL_CONFIG_NAME
    inference_model = WrappedModel.load_model(inference_model_json, inference_config=inference_config)
    inference_model.optimize_for_inference()

    dataset_with_predictions = run_inference_on_dataset(dataset=dataset_test, model=inference_model)

    # Save predictions to artifact folder
    path_predictions = artifact_folder_path / "predictions.jsonl"
    dataset_with_predictions.samples.write_ndjson(path_predictions)  # Json for readability

    no_gt_data = dataset_test.samples.select(pl.col(task_info.primitive.column_name()).list.len()).sum().item() == 0
    if no_gt_data:  # Skip metric calculation for tests sets without ground-truth annotations
        user_logger.warning("No ground-truth annotations found in the test set. Skipping metric calculation.")
        return logger

    metrics = metric_calculations(prediction_dataset=dataset_with_predictions)
    for metric_name, metric_value in metrics.items():
        logger.log_metric(metric_name, metric_value, step=0)

    return logger


def remove_images_with_no_bboxes(dataset: HafniaDataset, model_primitive: Type[Primitive]) -> HafniaDataset:
    if not dataset.has_primitive(model_primitive):
        raise ValueError("Dataset does not contain bounding box information.")

    filter_column_name = model_primitive.column_name()
    samples_with_bboxes = dataset.samples.filter(pl.col(filter_column_name).list.len() > 0)
    dataset = dataset.update_samples(samples_with_bboxes)
    return dataset


def get_dataset_task_from_model_primitive(
    dataset: HafniaDataset,
    model_primitive: str,
    task_name: Optional[str] = None,
) -> TaskInfo:
    """Logic to select dataset task that matches the model primitive"""

    # Get dataset tasks matching the model primitive
    matching_tasks = dataset.info.get_tasks_by_primitive(model_primitive)
    if len(matching_tasks) == 1:
        matching_task = matching_tasks[0]
        return matching_task

    if len(matching_tasks) == 0:
        available_primitives = [str(t.primitive.__name__) for t in dataset.info.tasks]
        raise ValueError(
            f"The selected model requires the dataset to have '{model_primitive}' annotations. "
            f"However, the dataset only contains the following primitives: {available_primitives}"
        )

    if task_name is None:
        matching_task_names = [t.name for t in matching_tasks]
        raise ValueError(
            f"The dataset contains multiple tasks with the required primitive '{model_primitive}'. "
            f"Please specify which task to use with the '--task_name' flag. "
            f"Matching tasks: {matching_task_names}"
        )

    model_task_info = dataset.info.get_task_by_name(task_name)

    if model_task_info.primitive != model_primitive:
        raise ValueError(f"The specified task '{task_name}' does not have the required primitive '{model_primitive}'.")

    return model_task_info


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
