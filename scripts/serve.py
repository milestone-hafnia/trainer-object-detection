import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import uvicorn
from cyclopts import App, Parameter
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hafnia.dataset.hafnia_dataset_types import ModelInfo
from hafnia.dataset.primitives import Primitive
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger
from PIL import Image
from pydantic import BaseModel, SerializeAsAny

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

"""
Launches model serving with FastAPI

# Launch the server with default RFDETRNano model and a threshold of 0.5:
python scripts/serve.py --model-path ./pretrained_models/RFDETRNano --inference.threshold 0.5

# Open a new terminal and test the health endpoint:
curl -s http://localhost:8080/ping
-> {"status":"Healthy"}

# Run prediction:
curl -s -X POST http://localhost:8080/predict -F "file=@./tests/43dd2464be75e57f27c090343b32da1b.jpg" | jq .
-> {
  "model_name": "RFDETRNano",
  "task_name": "object_detection",
  "predictions": [
    {
      "height": 0.8308719396591187,
      "width": 0.7169080376625061,
      "top_left_x": 0.06901198625564575,
      "top_left_y": 0.15795540809631348,
      "area": null,
      "class_name": "person",
      "class_idx": 1,
      "object_id": null,
      "confidence": 0.9593260884284973,
      "ground_truth": false,
      "task_name": "object_detection",
      "meta": null,
      "bboxes": null,
      "classifications": null,
      "polygons": null,
      "bitmasks": null
    },
    ...
  ]
}

"""
app = App(name="serve", help="Serve the trained object detection model over HTTP")

default_inference_config = InferenceConfig()


class PredictionResponse(BaseModel):
    model_name: str
    task_name: str
    predictions: List[SerializeAsAny[Primitive]]


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str


def _load_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(image)


def _build_app(model_path: str, inference_config: InferenceConfig) -> FastAPI:
    state: dict = {"model": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        path_model_config = Path(model_path) / "model_config.json"
        if not path_model_config.exists():
            raise FileNotFoundError(f"Could not find model config at: {path_model_config}")
        user_logger.info(f"Loading model from {path_model_config}")
        model = WrappedModel.load_model(path_model_config, inference_config=inference_config)
        model.optimize_for_inference()

        state["model"] = model
        user_logger.info("Model loaded and ready to serve")
        yield

    fastapi_app = FastAPI(title="Hafnia Object Detection Service", lifespan=lifespan)
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/info", response_model=ModelInfo)
    async def info():
        model: Optional[WrappedModel] = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return model.get_model_info()

    @fastapi_app.get("/ping", response_model=HealthResponse)
    async def ping():
        if state["model"] is None:
            return JSONResponse(status_code=503, content=HealthResponse(status="Model not loaded").model_dump())
        return HealthResponse(status="Healthy")

    @fastapi_app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        model: Optional[WrappedModel] = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            image = _load_image(await file.read())
            predictions = model.predict(image)

            return PredictionResponse(
                model_name=model.get_model_info().name,
                task_name=model.task.name,
                predictions=predictions,
            )
        except Exception as e:
            user_logger.exception("Prediction failed")
            return JSONResponse(status_code=500, content=ErrorResponse(error=str(e)).model_dump())

    @fastapi_app.post("/invocations", response_model=PredictionResponse)
    async def invocations(file: UploadFile = File(...)):
        return await predict(file)

    return fastapi_app


@app.default
def main(
    model_path: Annotated[str, Parameter(help="Path to trained model")] = "./pretrained_models/RFDETRNano",
    inference: Annotated[
        InferenceConfig, Parameter(help="Inference configuration for the model")
    ] = default_inference_config,
    host: Annotated[str, Parameter(help="Host interface to bind to")] = "127.0.0.1",
    port: Annotated[int, Parameter(help="Port to listen on")] = 8080,
):
    fastapi_app = _build_app(model_path=model_path, inference_config=inference)
    uvicorn.run(fastapi_app, host=host, port=port, timeout_keep_alive=300)


if __name__ == "__main__":
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
