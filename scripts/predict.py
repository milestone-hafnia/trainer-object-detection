from pathlib import Path

from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.primitives import Bbox
from hafnia.utils import progress_bar
from hafnia.visualizations import image_visualizations
from PIL import Image
from rfdetr import detr

from trainer_object_detection.train_utils import to_bbox_primitives

if __name__ == "__main__":
    dataset = HafniaDataset.from_name("midwest-vehicle-detection")
    path_prediction_visualization = Path(".data/predictions")
    path_prediction_visualization.mkdir(parents=True, exist_ok=True)

    model = detr.RFDETRNano(pretrain_weights=".data/models/checkpoint_best_ema.pth")

    dataset_split = dataset.create_split_dataset(split_name=SplitName.TEST)

    n_samples = 10
    test_subset = dataset_split.select_samples(n_samples=n_samples, seed=42)
    for i_sample, dict_sample in enumerate(progress_bar(test_subset)):
        sample = Sample(**dict_sample)

        image = sample.read_image()
        predictions = model.predict(image, threshold=0.35)

        task_info = dataset.info.get_task_by_primitive(Bbox)
        bboxes = to_bbox_primitives(predictions=predictions, sample=sample, class_names=task_info.class_names)

        annotations_visualized = image_visualizations.draw_annotations(image=image, primitives=bboxes)
        path_visualization = path_prediction_visualization / f"prediction_visualization_{i_sample}.png"
        Image.fromarray(annotations_visualized).save(path_visualization)
