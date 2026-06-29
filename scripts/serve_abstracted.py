from typing import Annotated, Optional

from cyclopts import App, Parameter
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger

from trainer_object_detection import utils
from trainer_object_detection.serving import run_inference_server
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="serve_abstracted", help="Serve an RF-DETR model over a REST API (abstracted serving)")

""" Serving examples
# Serve the default pretrained model and query it
python scripts/serve_abstracted.py --model-path ./pretrained_models/RFDETRNano.zip --inference.no-compile
curl localhost:8080/ping
curl --data-binary @tests/0e6d8275b955782b0cb8e9dafdebd086.png -H "Content-Type: image/jpeg" localhost:8080/invocations
curl -F "file=@tests/0e6d8275b955782b0cb8e9dafdebd086.png" localhost:8080/invocations
"""


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

    Functionally identical to ``scripts/serve.py``, but the serving itself is delegated to the
    reusable ``trainer_object_detection.serving`` layer: this script only keeps the boilerplate for
    configuring inference, resolving checkpoints and building the model, then hands the model - any
    object implementing the ``InferenceModel`` interface - to ``run_inference_server``. The serving
    layer is model-agnostic and is intended to move into the ``hafnia`` package later.
    """
    inference = inference or InferenceConfig()
    logger = HafniaLogger(project_name="Serve RF-DETR")

    # Prefer a user-selected checkpoint over the configured model when one is available.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' instead of '{model_path}'")
        model_path = checkpoint_model_path.as_posix()

    model = WrappedModel.load_model(model_path, inference_config=inference)
    model.optimize_for_inference()

    # The only serving code needed: pass any InferenceModel and it gets served.
    user_logger.info(f"Serving model on http://{host}:{port} (POST /invocations, GET /ping)")
    run_inference_server(model, host=host, port=port)


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
