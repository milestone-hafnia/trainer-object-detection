import argparse
import shutil
from pathlib import Path

import polars as pl
import torch
from hafnia import utils
from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.experiment import HafniaLogger
from hafnia.log import user_logger

from trainer_object_detection import train_utils


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "--dataset_local", type=str, default="midwest-vehicle-detection", help="Dataset being used locally"
    )
    parser.add_argument("--project_name", type=str, default="Trainer RF-DETR", help="Project name for the experiment")
    parser.add_argument("--model", type=str, default="RFDETRNano", help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Check cuda availability
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    logger = HafniaLogger(project_name=args.project_name)

    if utils.is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = utils.get_dataset_path_in_hafnia_cloud()  # The path to the full/hidden dataset is returned
        dataset = HafniaDataset.from_path(path_dataset)
    elif args.dataset_local:  # For local execution
        dataset = HafniaDataset.from_name(args.dataset_local)  # The small/public sample dataset is returned by name
    else:
        raise ValueError("You must provide a dataset name with the '--dataset_local DATASET_NAME' argument")

    args.dataset = dataset.info.dataset_name
    configuration = vars(args)
    configuration["has_cuda"] = has_cuda
    configuration["trainer"] = "DETR Object Detection"
    logger.log_configuration(configuration)  # Log the configuration to the UI

    dataset = remove_images_with_no_bboxes(dataset)

    model = train_utils.get_model_from_name(args.model, pretrain_weights=None)  # Get model from name

    # Convert dataset to COCO format for training
    dataset_name = dataset.info.dataset_name
    dataset_path = Path(".data") / f"format_coco_roboflow_{dataset_name}"
    dataset.to_coco_format(dataset_path)
    path_experiment = logger._local_experiment_path
    model.train(
        dataset_dir=dataset_path.as_posix(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_accumulation_steps=args.grad_accumulation_steps,
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
    args = parse_args()
    main(args)
