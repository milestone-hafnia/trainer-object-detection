import base64
import io
from typing import Annotated, Optional

import numpy as np
import uvicorn
from cyclopts import App, Parameter
from fastapi import FastAPI, HTTPException, Request
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from PIL import Image

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="serve", help="Serve an RF-DETR model over a REST API")

""" Serving examples
# Serve the default pretrained model and query it
python scripts/serve.py --model-path ./pretrained_models/RFDETRNano.zip --inference.no-compile
curl localhost:8080/ping
curl --data-binary @image.jpg -H "Content-Type: image/jpeg" localhost:8080/invocations
curl -F "file=@image.jpg" localhost:8080/invocations
"""

# The loaded model, populated by ``main`` before the server starts. The route handlers below
# read it directly - this script deliberately wires FastAPI by hand, with no serving abstraction.
# See ``scripts/serve_abstracted.py`` for the abstracted counterpart.
model: Optional[WrappedModel] = None

api = FastAPI(title="RF-DETR Inference Server")


def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB numpy array (matches the training/inference format)."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(image)


async def _read_images(request: Request) -> tuple[list[np.ndarray], bool]:
    """Read the request body into a list of images and an ``is_batch`` flag, based on its type.

    Supports a JSON body (``{"image": "<base64>"}`` or ``{"images": [...]}``), a
    ``multipart/form-data`` upload (one file or several), or a raw image body.
    """
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        payload = await request.json()
        if "images" in payload:
            return [_decode_image(base64.b64decode(item)) for item in payload["images"]], True
        if "image" in payload:
            return [_decode_image(base64.b64decode(payload["image"]))], False
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


@api.get("/ping")
def ping() -> dict:
    """SageMaker health check - a 200 response means the model is ready."""
    return {"status": "ok"}


@api.get("/model-info")
def model_info() -> dict:
    return model.get_model_info().model_dump(mode="json")


@api.post("/invocations")
async def invocations(request: Request) -> dict:
    """Run inference on the posted image(s) and return Hafnia primitives as JSON."""
    images, is_batch = await _read_images(request)

    # Predict one image at a time so each result keeps its own list of primitives.
    results = []
    for image in images:
        predictions = model.predict(image)
        height, width = image.shape[:2]
        results.append(
            {
                "predictions": [primitive.model_dump(mode="json") for primitive in predictions],
                "image": {"height": int(height), "width": int(width)},
            }
        )

    if is_batch:
        return {"results": results}
    return results[0]


@app.default
def main(
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to the model archive (.zip) to serve. Note: this is ignored when a checkpoint "
                "is available (e.g. a checkpoint selected for the experiment on the Hafnia platform) - "
                "the checkpoint is served instead of this model."
            )
        ),
    ] = "./pretrained_models/RFDETRNano.zip",
    inference: Annotated[Optional[InferenceConfig], Parameter(help="Inference configuration for the model")] = None,
    host: Annotated[str, Parameter(help="Host interface to bind the server to")] = "0.0.0.0",
    port: Annotated[int, Parameter(help="Port to serve on (SageMaker expects 8080)")] = 8080,
):
    """Serve an RF-DETR model over a SageMaker-compatible REST API.

    Loads the model from the compressed archive pointed to by ``model_path`` (or a user-selected
    checkpoint when one is available), optimizes it for inference, and starts an HTTP server exposing
    ``GET /ping`` (health check), ``GET /model-info`` and ``POST /invocations`` (prediction).
    ``POST /invocations`` accepts a raw image body (e.g. ``image/jpeg``), a ``multipart/form-data``
    file upload, or a JSON body with a base64-encoded ``image`` (or a list of ``images`` for batch
    inference), and returns predictions as serialized Hafnia primitives.

    This script wires FastAPI directly for simplicity; ``scripts/serve_abstracted.py`` does the same
    thing using the reusable serving layer in ``trainer_object_detection.serving``.
    """
    global model
    inference = inference or InferenceConfig()
    logger = HafniaLogger(project_name="Serve RF-DETR")

    # Prefer a user-selected checkpoint over the configured model when one is available.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' instead of '{model_path}'")
        model_path = checkpoint_model_path.as_posix()

    model = WrappedModel.load_model(model_path, inference_config=inference)
    model.optimize_for_inference()

    user_logger.info(f"Serving model on http://{host}:{port} (POST /invocations, GET /ping)")
    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
