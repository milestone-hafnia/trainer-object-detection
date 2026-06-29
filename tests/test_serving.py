import base64
import io
from typing import List, Union

import pytest
from fastapi.testclient import TestClient
from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.hafnia_dataset_types import ModelInfo, TaskInfo
from hafnia.dataset.primitives import Bbox, Primitive
from PIL import Image

from trainer_object_detection.serving import create_inference_app


class StubModel(InferenceModel):
    """Minimal InferenceModel returning one fixed Bbox, so serving can be tested without weights."""

    def predict(
        self,
        images: Union[ImageType, List[ImageType]],
        sample_dict: Union[dict, List[dict], None] = None,
    ) -> List[Primitive]:
        return [
            Bbox(
                top_left_x=0.1,
                top_left_y=0.2,
                width=0.3,
                height=0.4,
                class_idx=0,
                class_name="car",
                confidence=0.9,
                ground_truth=False,
            )
        ]

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="StubModel",
            tasks=[TaskInfo.from_class_names(primitive=Bbox, class_names=["car"])],
        )


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_inference_app(StubModel()))


def _png_bytes(size=(8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format="PNG")
    return buffer.getvalue()


def _assert_single_prediction(payload: dict, width: int = 8, height: int = 8):
    assert payload["image"] == {"height": height, "width": width}
    assert len(payload["predictions"]) == 1
    prediction = payload["predictions"][0]
    assert prediction["class_name"] == "car"
    assert prediction["ground_truth"] is False


def test_ping(client: TestClient):
    response = client.get("/ping")
    assert response.status_code == 200


def test_model_info(client: TestClient):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["name"] == "StubModel"


def test_invocations_raw_bytes(client: TestClient):
    response = client.post("/invocations", content=_png_bytes(), headers={"content-type": "image/png"})
    assert response.status_code == 200
    _assert_single_prediction(response.json())


def test_invocations_multipart(client: TestClient):
    response = client.post("/invocations", files={"file": ("image.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    _assert_single_prediction(response.json())


def test_invocations_json_base64(client: TestClient):
    encoded = base64.b64encode(_png_bytes()).decode("ascii")
    response = client.post("/invocations", json={"image": encoded})
    assert response.status_code == 200
    _assert_single_prediction(response.json())


def test_invocations_json_batch(client: TestClient):
    encoded = base64.b64encode(_png_bytes(size=(16, 8))).decode("ascii")
    response = client.post("/invocations", json={"images": [encoded, encoded]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    for result in results:
        _assert_single_prediction(result, width=16, height=8)


def test_invocations_multipart_batch(client: TestClient):
    files = [
        ("file", ("a.png", _png_bytes(), "image/png")),
        ("file", ("b.png", _png_bytes(), "image/png")),
    ]
    response = client.post("/invocations", files=files)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_invocations_missing_field_returns_400(client: TestClient):
    response = client.post("/invocations", json={"not_an_image": "x"})
    assert response.status_code == 400


def test_invocations_undecodable_image_returns_400(client: TestClient):
    response = client.post("/invocations", content=b"not an image", headers={"content-type": "image/png"})
    assert response.status_code == 400
