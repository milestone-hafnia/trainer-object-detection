# Trainer Package: Train Object Detection Model
This project demonstrates an object detection trainer package for Hafnia Training-as-a-Service (Training-aaS), compatible with object detection datasets such as "coco-2017" and "midwest-vehicle-detection".

Under the hood, this trainer package wraps the [RF-DETR](https://github.com/roboflow/rf-detr) model trainer by Roboflow. The training logic, model architecture, and core algorithms are provided by the upstream [`rfdetr`](https://pypi.org/project/rfdetr/) package — this repository adapts it to the Hafnia Training-aaS interface and dataset format. See [Acknowledgements](#acknowledgements), [License](#license), and [Citation](#citation) below.

> **Note:** This README covers the essential steps to get started. For more details on trainer packages and Training-aaS, visit the [trainer-classification README](https://github.com/milestone-hafnia/trainer-classification?tab=readme-ov-file#trainer-package-train-image-classification-model).

## Quick Start: No-Code Model Training

In this section, we will show how to launch model training using the Hafnia Training-aaS platform - without writing any code - using a pre-built trainer package.

### Steps:

1. **Access the Dashboard**  
   Navigate to the [experiments dashboard](https://hafnia.milestonesys.com/dashboard/training-aas/experiments) and click "Create Experiment"

2. **Select Dataset**  
   Choose your target dataset (e.g., `coco-2017` or `midwest-vehicle-detection`)

3. **Upload Trainer Package**  
   Download and upload the pre-built `trainer.zip` from: [trainer.zip](https://raw.githubusercontent.com/milestone-hafnia/trainer-object-detection/main/trainer.zip)

4. **Configure Training**  
   - **Training command:** `python scripts/train.py`
   - **Configuration:** Select "Free Tier" or "Professional" based on your needs

5. **Launch & Monitor**  
   Click "Create Experiment" and monitor progress in the dashboard

That's it! You have successfully launched an object detection model training experiment using the Hafnia Training-aaS platform.

For default training parameters, the trainer package converges in approximately 4 hours on the `midwest-vehicle-detection` dataset using the "Free Tier" configuration. 

---

# Trainer Package Development
In this section, we will guide on how to develop and run your own trainer package locally and in the hafnia platform.

## Setup and Install Trainer Package Locally
First, you need to clone the repository and install dependencies in a virtual environment using `uv` as the package manager.
```bash
# Download and install uv package manager on macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
cd [SOME_DESIRED_PATH]
git clone https://github.com/milestone-hafnia/trainer-object-detection

# Install dependencies in virtual environment using uv
cd trainer-object-detection
uv sync

source .venv/bin/activate
```

Now you have a virtual environment with all dependencies installed for the trainer package. This includes the Hafnia 
SDK and CLI which will be used in the following sections.

## Build Trainer Package Zip File
With the Hafnia CLI installed, you can now create your own trainer package using the Hafnia CLI.

```bash
# Update `trainer.zip` from command line
hafnia trainer create-zip .
```

This will create a `trainer.zip` file in the current folder, which can be uploaded to the Hafnia Web-portal as described 
in the Quick Start section. 

To validate that the trainer package works and that we will have no syntax or run time errors, you can run and debug the trainer package locally in VS Code. 

## (Optional) Run and Debug Trainer Package in VS Code
This trainer package is designed to work in a local environment with VS Code. To run and debug the trainer package in VS Code, follow these steps:

1. Open the project folder in VS Code through the IDE or by running `code .` from the terminal in the project folder.
2. Add the Python interpreter from the virtual environment `.venv/bin/python`.
   Press `Ctrl+Shift+P` and search for `Python: Select Interpreter`.
3. In the debug panel, select the configuration `Model Training` and press F5 or click the green play button 
   to start debugging. 


## Launch Experiment Directly from Command Line
The manual flow of packaging the trainer and uploading it through the Hafnia Web-portal becomes tedious when
running multiple experiments or making frequent updates to the trainer package.

To avoid this, you have the option of packaging and launching the trainer as an experiment directly using a single command. This
is demonstrated in the example below:

```bash
# First ensure that the Hafnia CLI is configured (Only done once)
hafnia configure

# Example 1: Package and launch experiment with default training command "python scripts/train.py"
hafnia experiment create --dataset midwest-detection-traffic --trainer-path .

# Example 2: Quick training
hafnia experiment create --dataset midwest-detection-traffic --trainer-path . --cmd "python scripts/train.py --epochs 1"

# Example 3: Package and launch experiment with custom training command
hafnia experiment create --dataset coco-2017 --trainer-path . --cmd "python scripts/train.py --model RFDETRSegPreview --batch_size 2  --epochs 3"

```
In above example the `--trainer-path` argument points to the local trainer package folder, the `midwest-vehicle-detection` dataset will be used in training. In the second example, we will use  override the default training command and add custom training parameters specific to the trainer package.

## Build and Launch Trainer Package Locally
Finally, this final section helps to debug your trainer package, if you get errors during the build phase. 

When a trainer package is launched in the Hafnia platform, it will first build your trainer package environment based on the `Dockerfile` and potentially other files in the trainer package. In this trainer package, the `Dockerfile` also uses `pyproject.toml`, `uv.lock` and `.python-version` to create a virtual environment with all dependencies installed for
your files. Once the build phase is complete, the trainer package will then be executed with the specified training command.

To simulate this process locally, you can use the Hafnia CLI to first build the Docker image from your `trainer.zip` file, and then launch the Docker image with a specified dataset. This is demonstrated in the example below:

```bash
# Create 'trainer.zip' from source folder
hafnia trainer create-zip .

# Build the Docker image locally from a 'trainer.zip' file
hafnia runc build-local trainer.zip

# Execute the Docker image locally with a desired dataset
hafnia runc launch-local --dataset midwest-vehicle-detection  "python scripts/train.py"
```

---

# Acknowledgements
This trainer package is a thin wrapper around [RF-DETR](https://github.com/roboflow/rf-detr) by [Roboflow](https://roboflow.com/). All credit for the underlying detection model, training procedure, and pretrained weights belongs to the RF-DETR authors. This repository merely adapts RF-DETR to the Hafnia Training-aaS interface — please refer to the upstream repository for questions about model behavior, training internals, and roadmap.

# License
This wrapper repository is released under the [MIT License](LICENSE).

The wrapped [`rfdetr`](https://github.com/roboflow/rf-detr) package and its Apache-designated model weights are distributed by Roboflow under the **Apache License 2.0**. Note that RF-DETR uses a split licensing model: the additional `rfdetr_plus` components and the RF-DETR-XL / 2XL detection models are licensed under **PML 1.0**, which has different terms (notably for commercial use). If you use those Plus components or weights via this trainer, you must comply with the PML 1.0 terms in addition to the Apache 2.0 terms that apply to the base package. Always consult the upstream [RF-DETR LICENSE](https://github.com/roboflow/rf-detr/blob/main/LICENSE) for authoritative terms.

# Citation
If you use this trainer package for research or publications, please cite the RF-DETR paper:

```bibtex
@misc{rf-detr,
    title={RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
    author={Isaac Robinson and Peter Robicheaux and Matvei Popov and Deva Ramanan and Neehar Peri},
    year={2025},
    eprint={2511.09554},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2511.09554},
}
```