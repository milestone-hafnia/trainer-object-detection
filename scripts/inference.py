from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from hafnia.dataset.benchmark.benchmark import run_inference_on_dataset
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset, Optional
from hafnia.dataset.primitives import PRIMITIVE_COLUMN_NAMES
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="inference", help="Run inference on a dataset")

CLASS_MAPPING_OPTIONS = [None, *utils.CLASS_MAPPINGS.keys()]

default_inference_config = InferenceConfig()


@app.default
def main(
    model_path: Annotated[str, Parameter(help=("Path to trained model"))] = "./pretrained_models/RFDETRNano",
    inference: Annotated[
        InferenceConfig, Parameter(help="Inference configuration for the model")
    ] = default_inference_config,
    model_class_mapping: Annotated[
        Optional[str],
        Parameter(help=f"Class mapping to use for the model. Options: {CLASS_MAPPING_OPTIONS}"),
    ] = None,
    split_name: Annotated[str, Parameter(help="Dataset split to run inference on")] = SplitName.TEST,
    output_path: Annotated[
        str, Parameter(help="Path where the dataset with predictions will be written")
    ] = "./.data/inference_output",
    samples: Annotated[
        Optional[int],
        Parameter(help="Limit the number of samples to run on. Useful for faster testing."),
    ] = None,
):

    logger = HafniaLogger(project_name="Inference RF-DETR")
    if is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = get_dataset_path_in_hafnia_cloud()  # The path to the full/hidden dataset is returned
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        # The small/public sample dataset is returned by name
        dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")

        # Drop all ground truth annotations
        dataset.samples = dataset.samples.drop(PRIMITIVE_COLUMN_NAMES, strict=False)

    path_model_config = Path(model_path) / "model_config.json"
    model = WrappedModel.load_model(path_model_config, inference_config=inference)
    model.optimize_for_inference()

    dataset_split = dataset.create_split_dataset(split_name=split_name)
    dataset_task_info = dataset.info.get_task_by_primitive(model.task.primitive)

    configuration = {
        "model": model.__class__.__name__,
        "compile": inference.compile,
        "batch_size": inference.batch_size,
        "threshold": inference.threshold,
        "dataset": dataset.info.dataset_name,
        "dataset_version": dataset.info.version,
        "model_filename": Path(model_path).name,
        "num_samples": len(dataset_split),
        "class_mapping_model": model_class_mapping,
        "split_name": split_name,
    }
    logger.log_configuration(configuration)

    if samples is not None:
        dataset_split = dataset_split.select_samples(n_samples=samples, seed=42)

    # Run inference on the dataset. Predictions are appended as new tasks on each sample.
    prediction_post_fix = "/predictions"
    dataset_with_predictions = run_inference_on_dataset(
        dataset=dataset_split,
        model=model,
        task_name_prediction_postfix=prediction_post_fix,
    )

    # Remap model prediction classes into the dataset's label space if requested
    if model_class_mapping is not None:
        class_mapping = utils.CLASS_MAPPINGS[model_class_mapping]
        dataset_with_predictions = dataset_with_predictions.class_mapper(
            class_mapping=class_mapping,
            method="remove_undefined",
            task_name=f"{dataset_task_info.name}{prediction_post_fix}",
        )

    path_output = Path(output_path)
    dataset_with_predictions.write(path_output)
    user_logger.info(f"Wrote dataset with predictions to: {path_output}")


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
