import hashlib
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest
import torch
from hafnia.experiment.command_builder import (
    DEFAULT_ORDER,
    CommandBuilderSchema,
    auto_save_command_builder_schema,
    path_of_function,
    simulate_form_data,
)

from trainer_object_detection.utils import CLI_TOOL


def file_hash(zip_file, name):
    """Get hash of the uncompressed file content inside a zip archive."""
    with zip_file.open(name) as f:
        return hashlib.md5(f.read()).hexdigest()


def compare_zip_files(zip_path1, zip_path2):
    files_changed = []
    with zipfile.ZipFile(zip_path1, "r") as z1, zipfile.ZipFile(zip_path2, "r") as z2:
        z1_files = sorted(z1.namelist())
        z2_files = sorted(z2.namelist())

        if z1_files != z2_files:
            print("The new trainer package contain new files")
            return False

        for name in z1_files:
            if file_hash(z1, name) != file_hash(z2, name):
                print(f"File content differs: {name}")
                files_changed.append(name)

    if len(files_changed) > 0:
        print(f"The following files have changed: {files_changed}")
        return False

    return True


def test_train_script():
    from scripts.train import main

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping integration test.")
    main(project_name="test_project", epochs=1, samples=40)


def test_benchmark_script():
    from scripts.benchmark import main

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping integration test.")
    main(samples=2, model_class_mapping="COCO2OnlyVehicle", dataset_class_mapping="Midwest2OnlyVehicle")


def test_predict_script():
    from scripts.visualize import main

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping integration test.")
    main(samples=2)


def test_export_onnx_script(tmp_path):
    from scripts.export_onnx import main

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping integration test.")
    logger = main()

    onnx_models = list(Path(logger.path_model()).glob("*.onnx"))
    assert len(onnx_models) > 0, "No ONNX models were exported."
    n_checkpoint_models = list(Path(logger.path_model_checkpoints()).glob("*.onnx"))
    assert len(n_checkpoint_models) > 0, "No ONNX models were exported to the checkpoints directory."


def test_export_openvino_script(tmp_path):
    import numpy as np
    import openvino as ov

    from scripts.export_openvino import main

    logger = main()

    openvino_xml_models = list(Path(logger.path_model()).glob("*.xml"))
    assert len(openvino_xml_models) > 0, "No OpenVINO IR (.xml) models were exported."
    openvino_bin_models = list(Path(logger.path_model()).glob("*.bin"))
    assert len(openvino_bin_models) > 0, "No OpenVINO IR (.bin) models were exported."

    n_checkpoint_xml_models = list(Path(logger.path_model_checkpoints()).glob("*.xml"))
    assert len(n_checkpoint_xml_models) > 0, "No OpenVINO IR (.xml) models were exported to the checkpoints directory."
    n_checkpoint_bin_models = list(Path(logger.path_model_checkpoints()).glob("*.bin"))
    assert len(n_checkpoint_bin_models) > 0, "No OpenVINO IR (.bin) models were exported to the checkpoints directory."

    # Beyond checking that files exist, verify the exported IR is actually usable: it should load,
    # compile, and run inference on CPU, producing outputs with valid shapes.
    core = ov.Core()
    for xml_path in openvino_xml_models:
        ov_model = core.read_model(xml_path)
        compiled_model = core.compile_model(ov_model, "CPU")

        # The model is exported with a dynamic batch dimension by default, so resolve any dynamic
        # dimensions (e.g. the batch size) to a concrete size of 1 to build a valid dummy input.
        partial_shape = ov_model.inputs[0].get_partial_shape()
        input_shape = [dim.get_length() if dim.is_static else 1 for dim in partial_shape]
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        outputs = compiled_model([dummy_input])

        assert len(outputs) > 0, f"No outputs were produced during inference for '{xml_path.name}'."
        for output in compiled_model.outputs:
            output_array = outputs[output]
            assert output_array.size > 0, f"Output '{output.get_any_name()}' is empty for '{xml_path.name}'."
            assert np.isfinite(output_array).all(), (
                f"Output '{output.get_any_name()}' contains non-finite values for '{xml_path.name}'."
            )

        # The pre-/post-processing recipe must be recoverable from the IR without external docs.
        assert ov_model.has_rt_info(["model_info"]), (
            f"Expected 'model_info' rt_info was not embedded in '{xml_path.name}'."
        )

        assert ov_model.has_rt_info(["model_info", "model_type"]), (
            f"Expected 'model_info/model_type' rt_info was not embedded in '{xml_path.name}'."
        )
        model_type = ov_model.get_rt_info(["model_info", "model_type"]).astype(str)
        assert model_type == "rfdetr", f"Unexpected model_type '{model_type}' in '{xml_path.name}'."

        assert ov_model.has_rt_info(["model_info", "labels"]), (
            f"Expected 'model_info/labels' rt_info was not embedded in '{xml_path.name}'."
        )
        labels = ov_model.get_rt_info(["model_info", "labels"]).astype(str).split(" ")
        assert len(labels) > 0, f"No class labels were embedded in '{xml_path.name}'."
        assert all(isinstance(label, str) and label for label in labels), (
            f"Class labels embedded in '{xml_path.name}' must be non-empty strings, got {labels}."
        )


class _StubLogger:
    """Minimal stand-in for ``HafniaLogger`` exposing only the checkpoints path."""

    def __init__(self, checkpoints_path):
        self._checkpoints_path = Path(checkpoints_path)

    def path_model_checkpoints(self):
        return self._checkpoints_path


def _make_checkpoint_zip(archive_path, class_names=("car", "truck")):
    """Build a checkpoint archive (model config + dummy weights) the way ``train.py`` does."""
    from hafnia.dataset.hafnia_dataset_types import TaskInfo
    from hafnia.dataset.primitives import Bbox

    from trainer_object_detection.wrapped_model import InitModelConfig

    archive_path = Path(archive_path)
    with tempfile.TemporaryDirectory() as source_dir:
        weights_path = Path(source_dir) / f"{archive_path.stem}.pth"
        weights_path.write_bytes(b"dummy-weights")
        model_config = InitModelConfig(
            name="RFDETRNano",
            task=TaskInfo.from_class_names(primitive=Bbox, class_names=list(class_names)),
            model_weight_path=str(weights_path),
        )
        model_config.save_model(archive_path)
    return archive_path


def test_get_checkpoint_if_available(tmp_path):
    """A checkpoint is discovered only when a ``*.zip`` archive is present, deterministically."""
    from trainer_object_detection.utils import get_checkpoint_if_available

    checkpoints_dir = tmp_path / "checkpoints"
    logger = _StubLogger(checkpoints_dir)

    # Missing checkpoints directory -> no checkpoint
    assert get_checkpoint_if_available(logger) is None

    # Empty directory -> no checkpoint
    checkpoints_dir.mkdir()
    assert get_checkpoint_if_available(logger) is None

    # Non-archive files are ignored
    (checkpoints_dir / "state.json").write_text("{}")
    assert get_checkpoint_if_available(logger) is None

    # A single checkpoint archive is returned
    _make_checkpoint_zip(checkpoints_dir / "checkpoint_best_ema.zip")
    assert get_checkpoint_if_available(logger) == checkpoints_dir / "checkpoint_best_ema.zip"

    # With multiple archives the selection is deterministic (sorted by name)
    _make_checkpoint_zip(checkpoints_dir / "checkpoint_best_regular.zip")
    assert get_checkpoint_if_available(logger) == checkpoints_dir / "checkpoint_best_ema.zip"


def test_checkpoint_is_loaded(tmp_path):
    """An available checkpoint is discovered and can be loaded back into a model config."""
    from trainer_object_detection.utils import get_checkpoint_if_available
    from trainer_object_detection.wrapped_model import InitModelConfig

    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    archive_path = _make_checkpoint_zip(
        checkpoints_dir / "checkpoint_best_ema.zip", class_names=["car", "bus", "truck"]
    )
    logger = _StubLogger(checkpoints_dir)

    checkpoint_model_path = get_checkpoint_if_available(logger)
    assert checkpoint_model_path == archive_path

    # The discovered checkpoint loads, with its weights extracted to an existing file on disk.
    model_config = InitModelConfig.load_model(checkpoint_model_path, use_weights=True)
    assert model_config.name == "RFDETRNano"
    assert [c.name for c in model_config.task.classes] == ["car", "bus", "truck"]
    assert Path(model_config.model_weight_path).exists()


def test_as_rgb_tensor(tmp_path):
    """Gray scale and RGBA images are converted to the (3, H, W) tensor expected by RF-DETR."""
    import numpy as np
    import torch
    from PIL import Image

    from trainer_object_detection.wrapped_model import as_rgb_tensor

    gray_scale = np.arange(6 * 4, dtype=np.uint8).reshape(6, 4)
    gray_scale_normalized = torch.from_numpy(gray_scale).float() / 255.0

    # Gray scale stored as (H, W) and as (H, W, 1) is replicated across the three channels
    for image in [gray_scale, gray_scale[:, :, None]]:
        image_tensor = as_rgb_tensor(image)
        assert image_tensor.shape == (3, 6, 4)
        assert image_tensor.dtype == torch.float32
        for i_channel in range(3):
            torch.testing.assert_close(image_tensor[i_channel], gray_scale_normalized)

    # RGB channels are kept as-is and the alpha channel of RGBA images is dropped
    rgb = np.random.randint(0, 256, size=(6, 4, 3), dtype=np.uint8)
    rgb_tensor = as_rgb_tensor(rgb)
    torch.testing.assert_close(rgb_tensor, torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    rgba = np.concatenate([rgb, np.full((6, 4, 1), 255, dtype=np.uint8)], axis=2)
    torch.testing.assert_close(as_rgb_tensor(rgba), rgb_tensor)

    # PIL images and image paths are handled as well - here for a gray scale ("L" mode) image
    image_pil = Image.fromarray(gray_scale)
    torch.testing.assert_close(as_rgb_tensor(image_pil), as_rgb_tensor(gray_scale))
    path_image = tmp_path / "gray_scale.png"
    image_pil.save(path_image)
    torch.testing.assert_close(as_rgb_tensor(path_image), as_rgb_tensor(gray_scale))

    # Values stay in the [0, 1] range required by RF-DETR
    assert 0.0 <= float(rgb_tensor.min()) and float(rgb_tensor.max()) <= 1.0

    with pytest.raises(ValueError):
        as_rgb_tensor(np.zeros((6, 4, 2), dtype=np.uint8))


class _StubDetections:
    """Minimal stand-in for a supervision ``Detections`` object holding a single full-image box."""

    def __init__(self, height: int, width: int):
        import numpy as np

        self.xyxy = np.array([[0, 0, width, height]], dtype=np.float32)
        self.class_id = np.array([0])
        self.confidence = np.array([0.9], dtype=np.float32)


class _StubRFDETR:
    """Stand-in for ``RFDETR`` that records the images it is given and mimics its return type."""

    def __init__(self):
        # Shapes are accumulated across calls, as 'predict_batch' predicts one image at a time.
        self.image_shapes = []

    def predict(self, images, threshold, include_source_image=True):
        import torch

        assert isinstance(images, list), "WrappedModel is expected to always predict on a list of images."
        assert all(isinstance(image, torch.Tensor) for image in images), (
            "Images should reach RF-DETR as tensors, so that no further conversion is needed."
        )
        self.image_shapes.extend(tuple(image.shape) for image in images)
        detections = [_StubDetections(height=image.shape[1], width=image.shape[2]) for image in images]
        # RF-DETR returns a bare 'Detections' object - not a list - when predicting on a single image.
        return detections if len(detections) > 1 else detections[0]


def _stub_wrapped_model():
    from hafnia.dataset.hafnia_dataset_types import TaskInfo
    from hafnia.dataset.primitives import Bbox

    from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

    task = TaskInfo.from_class_names(primitive=Bbox, class_names=["car"])
    return WrappedModel(model=_StubRFDETR(), task=task, inference_config=InferenceConfig())


def test_predict_single_gray_scale_image():
    """``predict`` takes a single image - gray scale included - and returns a flat prediction list."""
    import numpy as np

    gray_scale = np.zeros((8, 6), dtype=np.uint8)

    model = _stub_wrapped_model()
    predictions = model.predict(gray_scale)

    assert [p.class_name for p in predictions] == ["car"]
    assert model.model.image_shapes == [(3, 8, 6)]


def test_predict_batch_of_mixed_image_types(tmp_path):
    """``predict_batch`` takes a list of mixed image types and returns predictions per image.

    ``predict_batch`` is inherited from ``InferenceModel`` and calls ``predict`` once per image.
    """
    import numpy as np
    from PIL import Image

    gray_scale = np.zeros((8, 6), dtype=np.uint8)
    path_image = tmp_path / "gray_scale.png"
    Image.fromarray(gray_scale).save(path_image)
    images = [
        gray_scale,  # (H, W) gray scale array
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)),  # PIL RGB image
        path_image,  # path to a gray scale image
        np.zeros((4, 12, 4), dtype=np.uint8),  # RGBA array
    ]

    model = _stub_wrapped_model()
    predictions_per_image = model.predict_batch(images)

    assert len(predictions_per_image) == len(images)
    assert all(len(p) == 1 for p in predictions_per_image)

    # Every image reaches the model as a 3-channel tensor, in the original order
    assert model.model.image_shapes == [(3, 8, 6), (3, 10, 10), (3, 8, 6), (3, 4, 12)]

    # Each box is normalized by its own image shape, so the full-image stub box covers the whole image
    for image_predictions in predictions_per_image:
        assert image_predictions[0].width == pytest.approx(1.0)
        assert image_predictions[0].height == pytest.approx(1.0)


def _script_main(script_name: str):
    """Import the ``main`` function from a script module by name."""
    import importlib

    module = importlib.import_module(f"scripts.{script_name}")
    return module.main


@pytest.mark.parametrize("script_name", ["train", "benchmark", "export_onnx", "export_openvino"])
def test_command_builder_schema(script_name: str):
    """Test that the launch schema is up-to-date for each script."""
    main = _script_main(script_name)

    path_function = path_of_function(main)
    path_function_schema = path_function.with_suffix(".schema.json")

    if script_name == "train":
        order = 0
    else:
        order = DEFAULT_ORDER
    if not path_function_schema.exists():
        auto_save_command_builder_schema(main, cli_tool=CLI_TOOL, order=order)
        pytest.fail("Launch schema file not found. Schema file have been generated. Please run the test again.")

    actual_schema = CommandBuilderSchema.from_function(main, cli_tool=CLI_TOOL, order=order)
    current_schema = CommandBuilderSchema.from_json_file(path_function_schema)

    schema_is_up_to_date = current_schema == actual_schema
    assert schema_is_up_to_date, (
        f"Launch schema in '{path_function_schema}' is outdated. Please delete the schema file "
        f"({path_function_schema}) and rerun this test to regenerate it."
    )


def test_train_command_runs():
    """Test that the train script can be invoked end-to-end via the generated CLI args."""
    from scripts.train import main

    actual_schema = CommandBuilderSchema.from_function(main, cli_tool=CLI_TOOL)
    form_data = simulate_form_data(main, user_args={"stop_early": "True"})
    cmd_args = actual_schema.command_args_from_form_data(form_data)
    subprocess.run(cmd_args, shell=True, check=True)
