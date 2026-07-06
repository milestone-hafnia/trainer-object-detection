import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import openvino as ov
import torch
from cyclopts import App, Parameter
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="export_openvino", help="Export RF-DETR model to OpenVINO IR")

""" OpenVINO export examples

# Export the default pretrained model to OpenVINO IR
python scripts/export_openvino.py

# Export a trained checkpoint with a dynamic batch dimension
python scripts/export_openvino.py --model-path ./local_stuff/checkpoint_best_ema.zip --dynamic-batch

# Export with custom static resolution
python scripts/export_openvino.py --resolution 640
"""


# Kept internal on purpose: the CLI should expose the same arguments as export_onnx.py,
# except opset_version.
_ONNX_OPSET_VERSION = 17


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
    batch_size: Annotated[int, Parameter(help="Static batch size baked into the exported graph")] = 1,
    dynamic_batch: Annotated[
        bool,
        Parameter(help="Export with a dynamic batch dimension so the model accepts variable batch sizes at runtime"),
    ] = False,
    resolution: Annotated[
        Optional[int],
        Parameter(
            help=(
                "Input resolution (square side in pixels) baked into the exported graph. Defaults to the model's "
                "built-in resolution. Must be divisible by the backbone's patch_size * num_windows."
            )
        ),
    ] = None,
    backbone_only: Annotated[
        bool, Parameter(help="Export only the backbone (feature extractor) instead of the full detection model")
    ] = False,
    verbose: Annotated[bool, Parameter(help="Print export progress information")] = True,
):
    """Export an RF-DETR model archive to OpenVINO IR format.

    Loads the model from the compressed archive pointed to by ``model_path`` or from a user-selected
    checkpoint when one is available.

    The conversion pipeline is:

        RF-DETR checkpoint/archive -> temporary ONNX -> OpenVINO IR

    The temporary ONNX file is created only inside a temporary directory and is deleted after conversion.
    Only the resulting OpenVINO IR files, ``.xml`` and ``.bin``, are stored as Hafnia artifacts.

    The export options mirror the ONNX exporter where possible: ``batch_size`` bakes a static batch
    dimension into the graph, ``dynamic_batch`` enables variable batch size at runtime, ``resolution``
    overrides the square input size, and ``backbone_only`` exports just the feature extractor.
    """
    logger = HafniaLogger(project_name="Export RF-DETR OpenVINO")

    # Prefer a user-selected checkpoint over the configured model when one is available.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' instead of '{model_path}'")
        model_path = checkpoint_model_path.as_posix()

    # Load the model without optimize_for_inference / torch.compile.
    # ONNX export traces the raw model before OpenVINO conversion.
    wrapped_model = WrappedModel.load_model(model_path, inference_config=InferenceConfig())

    # RF-DETR places the model on CUDA by default; fall back to CPU so export also works locally.
    if not torch.cuda.is_available():
        user_logger.info("CUDA is not available. Exporting on CPU.")
        wrapped_model.model.model.device = torch.device("cpu")

    path_exported_checkpoints = logger.path_model_checkpoints()
    path_exported_models = logger.path_model()

    path_exported_checkpoints.mkdir(parents=True, exist_ok=True)
    path_exported_models.mkdir(parents=True, exist_ok=True)

    shape = (resolution, resolution) if resolution is not None else None

    configuration = {
        "model_filename": Path(model_path).name,
        "output_dir": path_exported_checkpoints.as_posix(),
        "batch_size": batch_size,
        "dynamic_batch": dynamic_batch,
        "resolution": resolution,
        "backbone_only": backbone_only,
    }
    logger.log_configuration(configuration)

    with tempfile.TemporaryDirectory(prefix="rfdetr_onnx_export_") as tmp_dir:
        tmp_onnx_dir = Path(tmp_dir)

        user_logger.info(f"Exporting temporary ONNX model to '{tmp_onnx_dir}'")

        wrapped_model.model.export(
            output_dir=tmp_onnx_dir.as_posix(),
            opset_version=_ONNX_OPSET_VERSION,
            batch_size=batch_size,
            dynamic_batch=dynamic_batch,
            shape=shape,
            backbone_only=backbone_only,
            verbose=verbose,
        )

        onnx_models = sorted(tmp_onnx_dir.rglob("*.onnx"))
        if not onnx_models:
            raise FileNotFoundError(f"No ONNX model was produced in temporary directory '{tmp_onnx_dir}'")

        for onnx_model_path in onnx_models:
            openvino_xml_path = path_exported_checkpoints / f"{onnx_model_path.stem}.xml"

            user_logger.info(f"Converting temporary ONNX model '{onnx_model_path.name}' to OpenVINO IR")

            openvino_model = ov.convert_model(onnx_model_path)
            ov.save_model(openvino_model, openvino_xml_path)

            user_logger.info(f"Saved OpenVINO IR to '{openvino_xml_path}'")

    # Store OpenVINO model as both a checkpoint and a model artifact.
    # OpenVINO IR consists of .xml and .bin files.
    for exported_file in path_exported_checkpoints.glob("*"):
        if exported_file.suffix.lower() not in {".xml", ".bin"}:
            continue

        destination = path_exported_models / exported_file.name
        shutil.copy2(exported_file, destination)
        user_logger.info(f"Copied exported OpenVINO model artifact to '{destination}'")

    return logger


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()