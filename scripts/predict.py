"""Static-benchmark inference entrypoint.

CLI contract matching the BENCH-1 static-benchmark SageMaker pipeline:

    python scripts/predict.py \
        --model-dir  $MDI_MODEL_DIR   \
        --dataset-dir $MDI_DATASET_DIR \
        --output-dir  $MDI_OUTPUT_DIR

Loads a bundled pretrained ``.zip`` from ``--model-dir``, runs RF-DETR inference on the
``HafniaDataset`` at ``--dataset-dir``, projects predictions into the shared 4-class ontology
via ``CLASS_MAPPINGS["COCO2FourClass"]``, and writes the resulting HafniaDataset (with a
``<task>/predictions`` task appended) to ``--output-dir``.

When running in a Hafnia cloud experiment (e.g. via ``hafnia experiment create --cmd 'python
scripts/predict.py ...'``), the pretrained ``.zip`` is also copied into ``HafniaLogger.path_model()``
so the finished dipdatalib Experiment record exposes a populated ``s3://`` ``model_url``. This
mirrors ``train.py``'s checkpoint-write pattern and gives ``predict.py`` a well-defined role as
the trainer's pretrained-model publication entrypoint (counterpart to ``train.py``'s fine-tuned-
model publication). Downstream ``compute_metrics.py`` (BENCH-1 phase 0-C) consumes
``--output-dir`` directly via ``HafniaDataset.from_path``.
"""
import shutil
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from hafnia.dataset.benchmark.benchmark import run_inference_on_dataset
from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from hafnia.utils import is_hafnia_cloud_job

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="predict", help="Static-benchmark inference: HafniaDataset in, predictions out.")

PREDICTION_POSTFIX = "/predictions"


@app.default
def main(
    model_dir: Annotated[
        Path,
        Parameter(help="Directory containing exactly one ``*.zip`` pretrained model archive."),
    ],
    dataset_dir: Annotated[
        Path,
        Parameter(help="Directory containing a HafniaDataset (as produced by ``write_annotations``)."),
    ],
    output_dir: Annotated[
        Path,
        Parameter(help="Directory to write the predictions-augmented HafniaDataset into."),
    ],
    project_name: Annotated[
        str,
        Parameter(help="Project name for the experiment (only used in cloud runs)."),
    ] = "Trainer RF-DETR",
):
    """Run inference and emit predictions as a HafniaDataset directory."""
    # D5: exactly one ``*.zip`` in --model-dir. Fail loudly on 0 or >1; do not auto-detect other layouts.
    model_zips = sorted(model_dir.glob("*.zip"))
    if len(model_zips) != 1:
        user_logger.error(
            f"--model-dir '{model_dir}' must contain exactly one '*.zip' archive; found {len(model_zips)}."
        )
        sys.exit(1)
    model_path = model_zips[0]

    model = WrappedModel.load_model(model_path, inference_config=InferenceConfig())
    model.optimize_for_inference()

    # Publish the pretrained archive as the experiment's artifact so dipdatalib records
    # a non-null ``s3://`` ``model_url`` on the finished experiment (D1). Skipped locally.
    if is_hafnia_cloud_job():
        logger = HafniaLogger(project_name=project_name)
        shutil.copy(model_path, logger.path_model() / model_path.name)
        user_logger.info(f"Published pretrained archive '{model_path.name}' to experiment path_model().")

    dataset = HafniaDataset.from_path(dataset_dir)
    dataset_task_info = dataset.info.get_task_by_primitive(model.task.primitive)

    # Predictions are appended as a new ``<task>/predictions`` task on each sample.
    dataset_with_predictions = run_inference_on_dataset(
        dataset=dataset,
        model=model,
        task_name_prediction_postfix=PREDICTION_POSTFIX,
    )

    # Project native COCO predictions into the shared 4-class ontology.
    dataset_with_predictions = dataset_with_predictions.class_mapper(
        class_mapping=utils.CLASS_MAPPINGS["COCO2FourClass"],
        method="remove_undefined",
        task_name=f"{dataset_task_info.name}{PREDICTION_POSTFIX}",
    )

    # Drop file_path so write_annotations serializes via its no-local-images branch
    if SampleField.FILE_PATH in dataset_with_predictions.samples.columns:
        dataset_with_predictions.samples = dataset_with_predictions.samples.drop(SampleField.FILE_PATH)

    # No metrics: predict.py is pure inference. Ground truth is not required.
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_with_predictions.write_annotations(output_dir)
    user_logger.info(f"Wrote predictions HafniaDataset to '{output_dir}'.")


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main' (matches train.py / benchmark.py)
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
