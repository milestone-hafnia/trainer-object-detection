import shutil
from pathlib import Path
from typing import Annotated, Optional

import torch
from cyclopts import App, Parameter
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="export_onnx", help="Export RF-DETR model to ONNX")

""" ONNX export examples
# Export the default pretrained model to ONNX
python scripts/export_onnx.py

# Export a trained checkpoint with a dynamic batch dimension and opset 19
python scripts/export_onnx.py --model-path ./local_stuff/checkpoint_best_ema.zip --dynamic-batch --opset-version 19
"""


@app.default
def main(
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to the model archive (.zip) to export. Note: this is ignored when a checkpoint "
                "is available (e.g. a checkpoint selected for the experiment on the Hafnia platform) - "
                "the checkpoint is exported instead of this model."
            )
        ),
    ] = "./pretrained_models/RFDETRNano.zip",
    opset_version: Annotated[int, Parameter(help="ONNX opset version to target")] = 17,
    batch_size: Annotated[int, Parameter(help="Static batch size baked into the ONNX graph")] = 1,
    dynamic_batch: Annotated[
        bool,
        Parameter(help="Export with a dynamic batch dimension so the model accepts variable batch sizes at runtime"),
    ] = False,
    resolution: Annotated[
        Optional[int],
        Parameter(
            help=(
                "Input resolution (square side in pixels) baked into the ONNX graph. Defaults to the model's "
                "built-in resolution. Must be divisible by the backbone's patch_size * num_windows."
            )
        ),
    ] = None,
    backbone_only: Annotated[
        bool, Parameter(help="Export only the backbone (feature extractor) instead of the full detection model")
    ] = False,
    verbose: Annotated[bool, Parameter(help="Print export progress information")] = True,
):
    """Export an RF-DETR model archive to ONNX format.

    Loads the model from the compressed archive pointed to by ``model_path`` (or a user-selected
    checkpoint when one is available) and exports it to ONNX via RF-DETR's built-in exporter.
    The resulting ``.onnx`` file is written to ``output_dir`` when provided, otherwise to the experiment model
    folder so it is collected as a model artifact on the Hafnia platform.

    The export options mirror RF-DETR's ``export`` API: ``opset_version`` selects the ONNX opset,
    ``batch_size`` bakes a static batch dimension into the graph (use ``dynamic_batch`` for a variable
    batch dimension instead), ``resolution`` overrides the square input size, and ``backbone_only``
    exports just the feature extractor.
    """
    logger = HafniaLogger(project_name="Export RF-DETR ONNX")

    # Prefer a user-selected checkpoint over the configured model when one is available.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' instead of '{model_path}'")
        model_path = checkpoint_model_path.as_posix()

    # Load the model without 'optimize_for_inference' (no torch.compile), as ONNX export traces the
    # raw model.
    # The inference settings (InferenceConfig) is required by WrappedModel but not used during export.
    wrapped_model = WrappedModel.load_model(model_path, inference_config=InferenceConfig())

    # RF-DETR places the model on CUDA by default; fall back to CPU so export also works locally.
    if not torch.cuda.is_available():
        user_logger.info("CUDA is not available. Exporting on CPU.")
        wrapped_model.model.model.device = torch.device("cpu")

    output_dir = logger.path_model_checkpoints().as_posix()

    shape = (resolution, resolution) if resolution is not None else None

    configuration = {
        "model_filename": Path(model_path).name,
        "output_dir": output_dir,
        "opset_version": opset_version,
        "batch_size": batch_size,
        "dynamic_batch": dynamic_batch,
        "resolution": resolution,
        "backbone_only": backbone_only,
    }
    logger.log_configuration(configuration)

    wrapped_model.model.export(
        output_dir=output_dir,
        opset_version=opset_version,
        batch_size=batch_size,
        dynamic_batch=dynamic_batch,
        shape=shape,
        backbone_only=backbone_only,
        verbose=verbose,
    )

    # To store model as both a checkpoint and a model artifact
    path_exported_models = logger.path_model()
    onnx_models = Path(output_dir).glob("*.onnx")
    for exported_file in onnx_models:
        shutil.copy2(exported_file, path_exported_models)
        user_logger.info(f"Copied exported model to '{path_exported_models / exported_file.name}'")

    return logger


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
