"""Reusable REST serving layer for Hafnia ``InferenceModel`` instances.

This module turns any model implementing the ``InferenceModel`` interface
(``predict`` + ``get_model_info``) into a SageMaker-compatible HTTP server, with no
knowledge of the concrete model. It is intentionally free of trainer-specific imports
so it can later be moved into the ``hafnia`` package unchanged.

The exposed endpoints follow the SageMaker real-time inference contract:
  - ``GET  /ping``        -> ``200`` health check.
  - ``GET  /model-info``  -> the model's ``ModelInfo`` as JSON.
  - ``POST /invocations`` -> run inference on the posted image(s).

``POST /invocations`` accepts three request encodings:
  - ``application/json``: ``{"image": "<base64>"}`` (single) or
    ``{"images": ["<base64>", ...]}`` (batch).
  - ``multipart/form-data``: one uploaded file (single) or several (batch).
  - any other content type (e.g. ``image/jpeg``, ``application/octet-stream``): the raw
    request body is treated as a single image.

Predictions are returned as serialized Hafnia primitives (Bbox / Bitmask / Polygon ...):
  - single -> ``{"predictions": [<primitive>, ...], "image": {"height": H, "width": W}}``
  - batch  -> ``{"results": [{"predictions": [...], "image": {...}}, ...]}``
"""

import base64
import io
from typing import List, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from hafnia.dataset.benchmark.inference_model import InferenceModel
from PIL import Image, UnidentifiedImageError


def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB ``np.ndarray`` of shape (H, W, 3), uint8.

    RGB matches the format produced by ``HafniaDataset`` image samples (PIL ->
    ``np.array``), which is what the models are trained and run on.
    """
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc
    return np.asarray(image)


def _decode_base64_image(encoded: str) -> np.ndarray:
    """Decode a base64-encoded image string into an RGB ``np.ndarray``."""
    try:
        data = base64.b64decode(encoded)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}") from exc
    return _decode_image(data)


async def _parse_images(request: Request) -> Tuple[List[np.ndarray], bool]:
    """Parse the request body into a list of images and a ``is_batch`` flag."""
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        payload = await request.json()
        if "images" in payload:
            return [_decode_base64_image(item) for item in payload["images"]], True
        if "image" in payload:
            return [_decode_base64_image(payload["image"])], False
        raise HTTPException(status_code=400, detail="JSON body must contain an 'image' or 'images' field.")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        # ``multi_items`` (not ``values``) keeps every part, including repeated field names.
        uploads = [value for _key, value in form.multi_items() if hasattr(value, "read")]
        if not uploads:
            raise HTTPException(status_code=400, detail="multipart/form-data body must contain at least one file.")
        images = [_decode_image(await upload.read()) for upload in uploads]
        return images, len(images) > 1

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Request body is empty.")
    return [_decode_image(body)], False


async def _predict_image(model: InferenceModel, image: np.ndarray) -> dict:
    """Run the model on a single image and return its serialized predictions."""
    # ``predict`` is a blocking (torch) call; run it off the event loop. We predict one
    # image at a time so each result keeps its own list of primitives (passing a list to
    # ``predict`` would flatten predictions across images).
    predictions = await run_in_threadpool(model.predict, image)
    height, width = image.shape[:2]
    return {
        "predictions": [primitive.model_dump(mode="json") for primitive in predictions],
        "image": {"height": int(height), "width": int(width)},
    }


def create_inference_app(model: InferenceModel) -> FastAPI:
    """Build a FastAPI app that serves ``model`` over the SageMaker inference contract."""
    app = FastAPI(title="Hafnia Inference Server")

    @app.get("/ping")
    def ping() -> dict:
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict:
        return model.get_model_info().model_dump(mode="json")

    @app.post("/invocations")
    async def invocations(request: Request) -> dict:
        images, is_batch = await _parse_images(request)
        results = [await _predict_image(model, image) for image in images]
        if is_batch:
            return {"results": results}
        return results[0]

    return app


def run_inference_server(model: InferenceModel, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Serve ``model`` over HTTP. Blocks until the server is stopped."""
    uvicorn.run(create_inference_app(model), host=host, port=port)
