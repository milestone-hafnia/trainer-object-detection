# The goal of a 'benchmark.py' script is to easily test the benchmark locally
# before running it in the Hafnia Cloud and users are more free to modify it.

# Model interfacing is placed on the user:
# - Code for model init, load and predict
#   - Implement model loading (ModelWrapper.__init__)
#   - Implementing ModelClass.predict:
#       - Pass image into model
#       - Convert model predictions into ModelPrediction
# - Virtual environment (Dockerfile + dependencies)

# Benchmark inputs and outputs
# Inputs:
# - Trainer Defined: Environment (Dockerfile + dependencies)
# - Trainer Defined: Model arguments
# - Trainer Defined: Model definition (source code)
# - Experiment Defined: Hardware - AWS instance
# - Experiment Defined: Benchmark dataset (HafniaDataset)
# - Experiment Defined (MISSING): Model weights
# - Experiment Defined (MISSING): Benchmark arguments which metrics to compute - or all!
# Outputs:
# - Benchmark metrics logged to HafniaLogger
#   - Should we store model predictions?
# - Model configuration? Model name, thresholds etc other hyperparameters?

import argparse

from hafnia import utils
from hafnia.dataset.hafnia_dataset import HafniaDataset

# The hafnia_benchmark should eventually be added to the hafnia package itself.
# User defined code
from trainer_object_detection.hafnia_benchmark import ModelInterface, benchmark_model


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--model", type=str, default="RFDETRNano", help="Model architecture to use")
    return parser.parse_args()


if __name__ == "__main__":
    model_args = parse_args()

    # Get benchmark dataset
    if utils.is_hafnia_cloud_job():  # For hafnia cloud execution (hidden dataset only)
        path_dataset = utils.get_dataset_path_in_hafnia_cloud()
        benchmark_dataset = HafniaDataset.from_path(path_dataset)
    else:  # For local execution (sample dataset only)
        benchmark_dataset = HafniaDataset.from_name("midwest-vehicle-detection")

    # Get model path
    if utils.is_hafnia_cloud_job():
        # path_model = utils.get_model_path_in_hafnia_cloud()  # Uncomment when the model library is supported
        path_model = ".data/models/checkpoint_best_ema.pth"
    else:  # For local execution (sample dataset only)
        path_model = ".data/models/checkpoint_best_ema.pth"

    model_wrapper = ModelInterface(model_config=model_args, path_model=path_model, dataset_info=benchmark_dataset.info)
    hyperparams = vars(model_args)

    benchmark_model(benchmark_dataset, model_wrapper, hyperparams)
