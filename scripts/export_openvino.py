import shutil
import tempfile
from pathlib import Path
from typing import Annotated, List, Optional, Sequence

import cv2
import nncf
import numpy as np
import openvino as ov
import torch
import torchvision.transforms.functional as F  # noqa: N812
from cyclopts import App, Parameter
from hafnia import utils as hafnia_utils
from hafnia.dataset import dataset_helpers
from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
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
python scripts/export_openvino.py --resolution 384

# Export and apply INT8 post-training quantization calibrated on a Hafnia dataset
python scripts/export_openvino.py --resolution 384 --quantize --calibration-dataset midwest-detection-traffic:1.0.0
"""


# Kept internal on purpose: the CLI should expose the same arguments as export_onnx.py,
# except opset_version.
_ONNX_OPSET_VERSION = 17

# Default local dataset used for calibration (matches the training dataset in train.py).
_DEFAULT_CALIBRATION_DATASET = "midwest-detection-traffic:1.0.0"

# Number of images sampled from the dataset to estimate activation ranges during quantization.
_CALIBRATION_SUBSET_SIZE = 300


def _load_calibration_hafnia_dataset(calibration_dataset: Optional[str]) -> HafniaDataset:
    """Load the dataset used for calibration.

    On the Hafnia platform the hidden dataset selected for the experiment is used (same as
    training). Locally, the public sample dataset is loaded by name - ``calibration_dataset``
    when provided, otherwise the training default. The name must include a version in the format
    ``name:version`` (e.g. ``midwest-vehicle-detection:2.0.0``).
    """
    if hafnia_utils.is_hafnia_cloud_job():
        path_dataset = hafnia_utils.get_dataset_path_in_hafnia_cloud()
        return HafniaDataset.from_path(path_dataset)

    dataset_ref = calibration_dataset or _DEFAULT_CALIBRATION_DATASET

    # Validate that the dataset reference includes a version
    if ":" not in dataset_ref:
        raise ValueError(
            f"Dataset version must be provided in the format 'name:version' "
            f"(e.g. 'midwest-vehicle-detection:2.0.0'), got '{dataset_ref}'"
        )

    dataset_name, dataset_version = dataset_helpers.dataset_name_and_version_from_string(dataset_ref)
    user_logger.info(f"Loading local calibration dataset '{dataset_name}' (version {dataset_version})")
    return HafniaDataset.from_name(dataset_name, version=dataset_version)


def _preprocess_image(
    image_bgr: np.ndarray,
    input_height: int,
    input_width: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Apply RF-DETR preprocessing to a BGR image and return an NCHW float32 tensor.

    Mirrors ``RFDETR.predict()`` (``to_tensor`` -> ``resize`` -> ``normalize``) using the same
    torchvision transforms and the model's own ``mean``/``std`` so calibration preprocessing stays
    in sync with inference and cannot drift.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(image_rgb)  # HWC uint8 RGB -> CHW float32 in [0, 1]
    tensor = F.resize(tensor, [input_height, input_width])
    tensor = F.normalize(tensor, list(mean), list(std))
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def _input_height_width(ov_model: ov.Model) -> tuple[int, int]:
    """Return the static (height, width) baked into the model's image input."""
    partial_shape = ov_model.inputs[0].get_partial_shape()
    if len(partial_shape) != 4 or not (partial_shape[2].is_static and partial_shape[3].is_static):
        raise ValueError(
            "Quantization requires a static spatial input shape, but the exported model input is "
            f"'{partial_shape}'. Export with a fixed '--resolution' to enable quantization."
        )
    return partial_shape[2].get_length(), partial_shape[3].get_length()


def _set_processing_rt_info(
    ov_model: ov.Model,
    mean: Sequence[float],
    std: Sequence[float],
    backbone_only: bool,
) -> None:
    """Embed the pre-/post-processing recipe into the model's ``rt_info``.

    Stores everything a consumer needs to reproduce RF-DETR preprocessing (channel order, resize,
    scaling, normalization) and to interpret the raw detection outputs, so the correct handling can
    be recovered later directly from the OpenVINO IR without external documentation.
    """
    # Preprocessing: BGR->RGB, resize to the model input, scale to [0, 1] (divide by 255),
    # then normalize with (x - mean) / std where mean/std are in the [0, 1] range.
    ov_model.set_rt_info(
        "RF-DETR preprocessing: convert BGR->RGB, resize to 'resize_hw', divide pixels by "
        "'scale_divisor' to get [0, 1], then normalize with (x - normalize_mean) / normalize_std.",
        ["preprocessing", "description"],
    )
    ov_model.set_rt_info("RGB", ["preprocessing", "color_order"])
    ov_model.set_rt_info(True, ["preprocessing", "reverse_input_channels"])
    ov_model.set_rt_info("NCHW", ["preprocessing", "layout"])
    ov_model.set_rt_info("bilinear", ["preprocessing", "resize_interpolation"])
    ov_model.set_rt_info(255.0, ["preprocessing", "scale_divisor"])
    ov_model.set_rt_info([float(v) for v in mean], ["preprocessing", "normalize_mean"])
    ov_model.set_rt_info([float(v) for v in std], ["preprocessing", "normalize_std"])

    partial_shape = ov_model.inputs[0].get_partial_shape()
    if len(partial_shape) == 4 and partial_shape[2].is_static and partial_shape[3].is_static:
        ov_model.set_rt_info(
            [partial_shape[2].get_length(), partial_shape[3].get_length()],
            ["preprocessing", "resize_hw"],
        )

    # Postprocessing only applies to the full detection model (the backbone emits raw features).
    if backbone_only:
        return

    ov_model.set_rt_info(
        "RF-DETR detection outputs: 'dets' = boxes as (cx, cy, w, h) normalized to [0, 1]; "
        "'labels' = per-class logits, apply sigmoid to obtain confidences.",
        ["postprocessing", "description"],
    )
    ov_model.set_rt_info("cxcywh_normalized", ["postprocessing", "dets", "box_format"])
    ov_model.set_rt_info("class_logits", ["postprocessing", "labels", "type"])
    ov_model.set_rt_info("sigmoid", ["postprocessing", "labels", "activation"])



def _build_calibration_dataset(
    hafnia_dataset: HafniaDataset,
    input_height: int,
    input_width: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> nncf.Dataset:
    """Build an NNCF calibration dataset from Hafnia dataset images."""
    split = hafnia_dataset.create_split_dataset(split_name=SplitName.TRAIN)
    n_samples = min(_CALIBRATION_SUBSET_SIZE, len(split))
    split = split.select_samples(n_samples=n_samples, seed=42)
    file_paths: List[str] = split.samples[SampleField.FILE_PATH].to_list()

    def transform_fn(file_path: str) -> np.ndarray:
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Could not read calibration image: '{file_path}'")
        return _preprocess_image(image, input_height, input_width, mean, std)

    return nncf.Dataset(file_paths, transform_fn)


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
    quantize: Annotated[
        bool,
        Parameter(
            help=(
                "Apply INT8 post-training quantization (NNCF) to the exported OpenVINO IR using a "
                "calibration dataset. Requires a static input resolution."
            )
        ),
    ] = False,
    calibration_dataset: Annotated[
        Optional[str],
        Parameter(
            help=(
                "Name of the Hafnia dataset used to calibrate INT8 quantization in the format "
                "'name:version' (e.g. 'midwest-vehicle-detection:2.0.0'). The version is required. "
                "Only used when '--quantize' is set and running locally (on the Hafnia platform the "
                "hidden experiment dataset is used instead). Defaults to "
                f"'{_DEFAULT_CALIBRATION_DATASET}'."
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

    When ``quantize`` is enabled, the converted OpenVINO IR is additionally quantized to INT8
    using NNCF post-training quantization. Calibration images are taken from the Hafnia dataset - the
    hidden experiment dataset on the Hafnia platform, or the dataset named by ``calibration_dataset``
    (defaulting to the training dataset) when running locally - and preprocessed exactly like during
    training. The dataset name must be provided in the format 'name:version' (e.g.,
    'midwest-detection-traffic:1.0.0'). Quantization requires a static input resolution (baked-in
    ``resolution``).
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

    if calibration_dataset is not None and not quantize:
        user_logger.warning(
            "'--calibration-dataset' is ignored because '--quantize' is not set."
        )

    configuration = {
        "model_filename": Path(model_path).name,
        "output_dir": path_exported_checkpoints.as_posix(),
        "batch_size": batch_size,
        "dynamic_batch": dynamic_batch,
        "resolution": resolution,
        "backbone_only": backbone_only,
        "quantize": quantize,
        "calibration_dataset": calibration_dataset if quantize else None,
    }
    logger.log_configuration(configuration)

    # Load the calibration dataset up-front so failures happen before the (slow) export step.
    hafnia_dataset = _load_calibration_hafnia_dataset(calibration_dataset) if quantize else None

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
            model_stem = f"{onnx_model_path.stem}_int8" if quantize else onnx_model_path.stem
            openvino_xml_path = path_exported_checkpoints / f"{model_stem}.xml"

            user_logger.info(f"Converting temporary ONNX model '{onnx_model_path.name}' to OpenVINO IR")

            openvino_model = ov.convert_model(onnx_model_path)

            if quantize:
                assert hafnia_dataset is not None  # guaranteed when quantization is enabled
                input_height, input_width = _input_height_width(openvino_model)
                user_logger.info(
                    f"Quantizing '{onnx_model_path.name}' to INT8 using up to "
                    f"{_CALIBRATION_SUBSET_SIZE} calibration images at {input_width}x{input_height}"
                )
                nncf_calibration_dataset = _build_calibration_dataset(
                    hafnia_dataset,
                    input_height=input_height,
                    input_width=input_width,
                    mean=wrapped_model.model.means,
                    std=wrapped_model.model.stds,
                )
                # Cap the subset size to the images actually available to avoid an NNCF warning.
                subset_size = nncf_calibration_dataset.get_length() or _CALIBRATION_SUBSET_SIZE
                openvino_model = nncf.quantize(
                    openvino_model,
                    nncf_calibration_dataset,
                    subset_size=subset_size,
                )

            # Record the pre-/post-processing recipe in the IR so it can be recovered later.
            _set_processing_rt_info(
                openvino_model,
                mean=wrapped_model.model.means,
                std=wrapped_model.model.stds,
                backbone_only=backbone_only,
            )

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