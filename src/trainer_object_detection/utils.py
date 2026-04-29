import types

import mlflow

CLI_TOOL = "cyclopts"

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

_METRIC_KEY_MAP = {
    "train/loss": "Loss/Train",
    "val/loss": "Loss/Test",
    "val/mAP_50_95": "Metrics/Base/AP50_90",
    "val/mAP_50": "Metrics/Base/AP50",
    "val/mAR": "Metrics/Base/AR50_90",
    "val/ema_mAP_50_95": "Metrics/EMA/AP50_90",
    "val/ema_mAP_50": "Metrics/EMA/AP50",
    "val/ema_mAR": "Metrics/EMA/AR50_90",
}


def patch_to_support_experiment_tracker_with_hafnia(detr: types.ModuleType):
    import rfdetr.training as _rfdetr_training
    from pytorch_lightning import Callback

    class _HafniaMLflowCallback(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            for ptl_key, hafnia_key in _METRIC_KEY_MAP.items():
                value = trainer.callback_metrics.get(ptl_key)
                if value is not None:
                    mlflow.log_metric(hafnia_key, float(value), step=epoch)

    _original_build_trainer = _rfdetr_training.build_trainer

    def _patched_build_trainer(train_config, model_config, **kwargs):
        ptl_trainer = _original_build_trainer(train_config, model_config, **kwargs)
        ptl_trainer.callbacks.append(_HafniaMLflowCallback())
        return ptl_trainer

    _rfdetr_training.build_trainer = _patched_build_trainer
    return detr
