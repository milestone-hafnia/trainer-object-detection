from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hafnia.dataset.hafnia_dataset_types import ModelInfo
from hafnia.dataset.primitives import Bbox

from scripts.serve import _build_app
from trainer_object_detection.wrapped_model import InferenceConfig

PATH_REPO = Path(__file__).parents[1]
PATH_MODEL = PATH_REPO / "pretrained_models" / "RFDETRNano"
PATH_IMAGE = Path(__file__).parent / "43dd2464be75e57f27c090343b32da1b.jpg"


@pytest.fixture(scope="module")
def client():
    fastapi_app = _build_app(model_path=str(PATH_MODEL), inference_config=InferenceConfig(threshold=0.5))
    with TestClient(fastapi_app) as test_client:
        yield test_client


def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy"}


def test_info(client):
    response = client.get("/info")
    assert response.status_code == 200, response.text
    body = response.json()

    model = ModelInfo.model_validate(body)  # Validate response schema
    assert model.name == "RFDETRNano"

    tasks = model.tasks
    assert isinstance(tasks, list) and len(tasks) > 0
    task = tasks[0]
    assert task.name, "Task should have a name"
    assert task.primitive is Bbox, "Task should declare a primitive"

    classes = task.get_class_names()
    assert isinstance(classes, list) and len(classes) > 0, "Task should expose its class list"


def test_predict(client):
    with PATH_IMAGE.open("rb") as f:
        response = client.post("/predict", files={"file": (PATH_IMAGE.name, f, "image/jpeg")})

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["model_name"] == "RFDETRNano"
    assert isinstance(body["task_name"], str) and body["task_name"]

    predictions = body["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) > 0, "Expected at least one detection on the test image"

    bbox = predictions[0]
    for field in ("top_left_x", "top_left_y", "width", "height", "class_idx", "class_name", "confidence"):
        assert field in bbox, f"Missing field '{field}' — primitive serialization is broken"

    for value in (bbox["top_left_x"], bbox["top_left_y"], bbox["width"], bbox["height"]):
        assert 0.0 <= value <= 1.0, f"Bbox coordinate out of normalized range: {value}"

    assert bbox["confidence"] >= 0.5
