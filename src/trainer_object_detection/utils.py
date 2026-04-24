import types

import mlflow

# TYPE_MODEL = Literal["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase", "RFDETRLarge", "RFDETRSegPreview"]
CLI_TOOL = "cyclopts"


def safe_index(arr, idx):
    return arr[idx] if 0 <= idx < len(arr) else None


CLASS_MAPPINGS = {
    "COCO2OnlyVehicle": {
        "bicycle": "Vehicle",
        "car": "Vehicle",
        "motorcycle": "Vehicle",
        "bus": "Vehicle",
        "truck": "Vehicle",
        # Ignore the remaining classes
    },
    "Midwest2OnlyVehicle": {
        "Vehicle*": "Vehicle",
        # Ignore the remaining classes
    },
}

# NOTE: Remove COCO_CLASSES her and use 'rfdetr.assets.coco_classes.COCO_CLASSES' for new versions of rfdetr.
COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

COCO_CLASS_NAMES: list[str] = [name for _, name in sorted(COCO_CLASSES.items())]


class MetricsTensorBoardSinkMLflow:
    """
    Replacement for MetricsTensorBoardSink that logs to MLflow instead of TensorBoard.
    Keeps the same interface: __init__, update, close.
    """

    def __init__(self, output_dir: str):
        print("MLflow Metrics sink initialized")

    def update(self, values: dict):
        epoch = values["epoch"]

        # losses
        if "train_loss" in values:
            mlflow.log_metric("Loss/Train", values["train_loss"], step=epoch)
        if "test_loss" in values:
            mlflow.log_metric("Loss/Test", values["test_loss"], step=epoch)

        # standard COCO eval
        if "test_coco_eval_bbox" in values:
            coco_eval = values["test_coco_eval_bbox"]
            ap50_90 = safe_index(coco_eval, 0)
            ap50 = safe_index(coco_eval, 1)
            ar50_90 = safe_index(coco_eval, 8)
            if ap50_90 is not None:
                mlflow.log_metric("Metrics/Base/AP50_90", ap50_90, step=epoch)
            if ap50 is not None:
                mlflow.log_metric("Metrics/Base/AP50", ap50, step=epoch)
            if ar50_90 is not None:
                mlflow.log_metric("Metrics/Base/AR50_90", ar50_90, step=epoch)

        # EMA COCO eval
        if "ema_test_coco_eval_bbox" in values:
            ema_coco_eval = values["ema_test_coco_eval_bbox"]
            ema_ap50_90 = safe_index(ema_coco_eval, 0)
            ema_ap50 = safe_index(ema_coco_eval, 1)
            ema_ar50_90 = safe_index(ema_coco_eval, 8)
            if ema_ap50_90 is not None:
                mlflow.log_metric("Metrics/EMA/AP50_90", ema_ap50_90, step=epoch)
            if ema_ap50 is not None:
                mlflow.log_metric("Metrics/EMA/AP50", ema_ap50, step=epoch)
            if ema_ar50_90 is not None:
                mlflow.log_metric("Metrics/EMA/AR50_90", ema_ar50_90, step=epoch)

    def save(self):
        pass

    def close(self):
        pass


def patch_to_support_experiment_tracker_with_hafnia(detr: types.ModuleType):
    detr.MetricsPlotSink = MetricsTensorBoardSinkMLflow

    return detr
