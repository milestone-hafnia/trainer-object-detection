from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from hafnia.dataset import image_visualizations
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job, progress_bar
from PIL import Image

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="predict", help="Run prediction and save visualizations")

default_inference_config = InferenceConfig(compile=True, batch_size=1, threshold=0.35)


@app.default
def main(
    model_path: Annotated[str, Parameter(help="Path to trained model")] = "./pretrained_models/RFDETRNano",
    inference: Annotated[
        InferenceConfig, Parameter(help="Inference configuration for the model")
    ] = default_inference_config,
    output_path: Annotated[
        str, Parameter(help="Directory where prediction visualizations are saved")
    ] = ".data/predictions",
    split_name: Annotated[str, Parameter(help="Dataset split to run prediction on")] = SplitName.TEST,
    samples: Annotated[int, Parameter(help="Number of samples to predict and visualize")] = 10,
):
    path_prediction_visualization = Path(output_path)
    path_prediction_visualization.mkdir(parents=True, exist_ok=True)

    if is_hafnia_cloud_job():  # For hafnia cloud execution
        path_dataset = get_dataset_path_in_hafnia_cloud()  # The path to the full/hidden dataset is returned
        dataset = HafniaDataset.from_path(path_dataset)
    else:
        # The small/public sample dataset is returned by name
        dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")

    path_model_config = Path(model_path) / "model_config.json"
    model = WrappedModel.load_model(path_model_config, inference_config=inference)
    model.optimize_for_inference()

    dataset_split = dataset.create_split_dataset(split_name=split_name)

    test_subset = dataset_split.select_samples(n_samples=samples, seed=42)
    for i_sample, dict_sample in enumerate(progress_bar(test_subset)):
        sample = Sample(**dict_sample)

        image = sample.read_image()
        predictions = model.predict(image)
        annotations_visualized = image_visualizations.draw_annotations(image=image, primitives=predictions)
        path_visualization = path_prediction_visualization / f"prediction_visualization_{i_sample}.png"
        Image.fromarray(annotations_visualized).save(path_visualization)


if __name__ == "__main__":
    app()
