from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from hafnia.dataset.benchmark.benchmark import run_benchmark
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.dataset_recipe.recipe_transforms import ClassMapper
from hafnia.dataset.hafnia_dataset import HafniaDataset, Optional
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="benchmark", help="Benchmark")

CLASS_MAPPING_OPTIONS = [None, *utils.CLASS_MAPPINGS.keys()]

default_inference_config = InferenceConfig()

""" Benchmarking examples
# Example: Benchmark pretrained model (RFDETRNano) for a vehicle detection task.
# The tricky part for this benchmark is that RFDETRNano is pretrained on coco datasets lables while the
# dataset have different labels. To solve this we needs to remap both the dataset and model predictions to
# a common label space. In this example we remap both to a common "vehicle detection"
# label space, but other remapping strategies are also possible.
python scripts/benchmark.py --model-class-mapping COCO2OnlyVehicle


hafnia experiment create --recipe-id 8618234d-b4da-4aa9-bb3e-3be86bb50369 --trainer-path . --cmd "python scripts/benchmark.py --model-class-mapping COCO2OnlyVehicle"

"""


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
    dataset_class_mapping: Annotated[
        Optional[str],
        Parameter(help=f"Class mapping to use for the dataset ground truth. Options: {CLASS_MAPPING_OPTIONS}"),
    ] = None,
    samples: Annotated[
        Optional[int],
        Parameter(help="Limit the number of samples to run on. Useful for faster testing."),
    ] = None,
):

    logger = HafniaLogger(project_name="Benchmarking RF-DETR")
    if is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = get_dataset_path_in_hafnia_cloud()  # The path to the full/hidden dataset is returned
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        # The small/public sample dataset is returned by name
        dataset = HafniaDataset.from_name("coco-2017", version="1.0.0")

    path_model_config = Path(model_path) / "model_config.json"
    model = WrappedModel.load_model(path_model_config, inference_config=inference)
    model.optimize_for_inference()

    dataset_split = dataset.create_split_dataset(split_name=SplitName.TEST)
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
        "class_mapping_dataset": dataset_class_mapping,
    }
    logger.log_configuration(configuration)

    if samples is not None:
        dataset_split = dataset_split.select_samples(n_samples=samples, seed=42)

    # Remapping of model prediction and/or ground-truth classes to a common label space
    prediction_post_fix = "/predictions"
    recipe_transforms = []
    if model_class_mapping is not None:
        recipe_transforms.append(
            ClassMapper(
                class_mapping=utils.CLASS_MAPPINGS[model_class_mapping],
                method="remove_undefined",
                task_name=f"{dataset_task_info.name}{prediction_post_fix}",
            )
        )
    if dataset_class_mapping is not None:
        dataset_split = dataset_split.class_mapper(
            class_mapping=utils.CLASS_MAPPINGS[dataset_class_mapping],
            method="remove_undefined",
            task_name=model.task.name,
        )

    metrics, _ = run_benchmark(
        dataset=dataset_split,
        model=model,
        recipe_transforms=recipe_transforms,
        task_name_prediction_postfix=prediction_post_fix,
    )

    for metric_name, metric_value in metrics.items():
        logger.log_metric(metric_name, metric_value, step=0)


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
