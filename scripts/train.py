import shutil
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
import torch
from cyclopts import App, Parameter
from hafnia import utils
from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives import Bitmask
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from rfdetr import detr

from trainer_object_detection.train_utils import patch_to_support_experiment_tracker_with_hafnia

detr = patch_to_support_experiment_tracker_with_hafnia(detr)

CLI_TOOL = "cyclopts"

TYPE_MODEL = Literal["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase", "RFDETRLarge", "RFDETRSegPreview"]


app = App(name="train", help="PyTorch Training")


@app.default
def main(
    project_name: Annotated[str, Parameter(help="Project name for the experiment")] = "Trainer RF-DETR",
    model: Annotated[TYPE_MODEL, Parameter(help="Model architecture to use")] = "RFDETRNano",
    epochs: Annotated[int, Parameter(help="Number of epochs to train")] = 10,
    batch_size: Annotated[int, Parameter(help="Batch size for training")] = 8,
    grad_accumulation_steps: Annotated[int, Parameter(help="Gradient accumulation steps")] = 1,
    learning_rate: Annotated[float, Parameter(help="Learning rate for optimizer")] = 0.001,
    stop_early: Annotated[
        bool,
        Parameter(help="Break script before training starts. Can be used to avoid long training times during testing."),
    ] = False,
):
    # Check cuda availability
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    logger = HafniaLogger(project_name=project_name)

    if utils.is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = utils.get_dataset_path_in_hafnia_cloud()  # The path to the full/hidden dataset is returned
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        # # The small/public sample dataset is returned by name
        dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")

    configuration = {
        "model": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "learning_rate": learning_rate,
        "dataset": dataset.info.dataset_name,
        "has_cuda": has_cuda,
    }
    logger.log_configuration(configuration)

    if model == "RFDETRNano":
        model = detr.RFDETRNano()
    elif model == "RFDETRSmall":
        model = detr.RFDETRSmall()
    elif model == "RFDETRMedium":
        model = detr.RFDETRMedium()
    elif model == "RFDETRBase":
        model = detr.RFDETRBase()
    elif model == "RFDETRLarge":
        model = detr.RFDETRLarge()
    elif model == "RFDETRSegPreview":
        torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid runtime errors with RFDETRSegPreview
        model = detr.RFDETRSegPreview()
    else:
        raise ValueError(f"Model {model} not recognized.")

    if isinstance(model, detr.RFDETRSegPreview) and not dataset.has_primitive(Bitmask):
        raise ValueError(
            "You have selected an instance segmentation model ('RFDETRSegPreview') which requires a dataset with "
            f"'Bitmask' primitive. The selected dataset '{dataset.info.dataset_name}' does not "
            "include 'Bitmask' primitives. Please select a different model or dataset."
        )
    dataset = remove_images_with_no_bboxes(dataset)

    # Convert dataset to COCO format for training
    dataset_name = dataset.info.dataset_name
    dataset_path = Path(".data") / f"format_coco_roboflow_{dataset_name}"
    dataset.to_coco_format(dataset_path)
    path_experiment = logger._local_experiment_path
    path_experiment.mkdir(parents=True, exist_ok=True)

    if stop_early:
        user_logger.info("Early stopping before training was activated with '--stop_early' flag.")
        return None

    model.train(
        dataset_dir=dataset_path.as_posix(),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        grad_accumulation_steps=grad_accumulation_steps,
        output_dir=path_experiment.as_posix(),
    )

    # Move final model weights to model folder (e.g. "checkpoint_best_regular.pth" and "checkpoint_best_total.pth")
    model_paths = list(path_experiment.glob("checkpoint_*.pth"))
    model_folder_path = logger.path_model()  # Store model here to make it available in the UI.
    for model_path in model_paths:
        shutil.copy2(model_path, model_folder_path)

    # Move checkpoints to checkpoints folder (e.g. "checkpoint0000.pth", "checkpoint0010.pth")
    # (Both models and checkpoints start with "checkpoint", so we exclude 'model_paths' from checkpoint models)
    checkpoint_model_paths = set(path_experiment.glob("checkpoint*.pth")) - set(model_paths)
    checkpoints_folder_path = logger.path_model_checkpoints()
    for ckpt_path in checkpoint_model_paths:
        shutil.copy2(ckpt_path, checkpoints_folder_path)

    # Move files to artifact folder
    artifact_folder_path = logger._path_artifacts()
    check_for_files = ["log.txt", "metrics_plot.png", "events.out.tfevents*", "results.json"]
    for file_pattern in check_for_files:
        file_paths = list(path_experiment.glob(file_pattern))
        if len(file_paths) == 0:
            user_logger.warning(f"No files found for pattern: {file_pattern}")
        for file_path in file_paths:
            shutil.copy2(file_path, artifact_folder_path)
    return logger


def remove_images_with_no_bboxes(dataset: HafniaDataset) -> HafniaDataset:
    # Remove images with no bounding boxes to avoid runtime errors during training
    if SampleField.BITMASKS in dataset.samples.columns:
        filter_column_name = SampleField.BITMASKS
    elif SampleField.BBOXES in dataset.samples.columns:
        filter_column_name = SampleField.BBOXES
    else:
        raise ValueError("Dataset does not contain bounding box information.")
    samples_with_bboxes = dataset.samples.filter(pl.col(filter_column_name).list.len() > 0)
    dataset = dataset.update_samples(samples_with_bboxes)
    return dataset


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
