import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
import torch
from hafnia import utils
from hafnia.experiment.command_builder import (
    CommandBuilderSchema,
    auto_save_command_builder_schema,
    path_of_function,
    simulate_form_data,
)
from hafnia.platform.builder import validate_trainer_package_format


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


def test_trainer_zip_outdated(tmp_path: Path):
    """Test the trainer package generation and validation."""
    path_trainer_zip_actual = tmp_path / "trainer.zip"
    path_trainer_zip_expected = Path(__file__).parents[1] / "trainer.zip"
    path_source = Path("./.")
    utils.archive_dir(path_source, output_path=path_trainer_zip_actual)
    validate_trainer_package_format(path_trainer_zip_actual)

    if not path_trainer_zip_expected.exists():
        shutil.copy2(path_trainer_zip_actual, path_trainer_zip_expected)
        assert 0 == 1, "Trainer package zip file not found. Package have been regenerated. Please run the test again."

    assert_msg = (
        "Trainer package contents differ. Please check the differences. "
        f"Delete the '{path_trainer_zip_expected}' file to regenerate it or "
        "run 'hafnia trainer create-zip .' in terminal to update the recipe."
    )
    assert compare_zip_files(path_trainer_zip_actual, path_trainer_zip_expected), assert_msg


def test_integration_test_placeholder():
    from scripts.train import main

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping integration test.")
    main(project_name="test_project", epochs=1)


def test_command_builder_schema():
    """Test that the launch schema can be saved to a file."""
    from scripts.train import CLI_TOOL, main

    path_function = path_of_function(main)
    path_function_schema = path_function.with_suffix(".schema.json")

    if not path_function_schema.exists():
        auto_save_command_builder_schema(main, cli_tool=CLI_TOOL)
        pytest.fail("Launch schema file not found. Schema file have been generated. Please run the test again.")

    actual_schema = CommandBuilderSchema.from_function(main, cli_tool=CLI_TOOL)
    current_schema = CommandBuilderSchema.from_json_file(path_function_schema)

    schema_is_up_to_date = current_schema == actual_schema
    assert schema_is_up_to_date, (
        f"Launch schema in '{path_function_schema}' is outdated. Please delete the schema file "
        f"({path_function_schema}) and rerun this test to regenerate it."
    )

    form_data = simulate_form_data(main, user_args={"stop_early": "True"})
    cmd_args = actual_schema.command_args_from_form_data(form_data)
    subprocess.run(cmd_args, shell=True, check=True)
