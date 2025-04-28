# Taken from
# https://github.com/cambridge-mlg/sgd-gp/blob/main/scalable_gps/configs/default.py

DOCKSTRING_INPUT_DIM = 1024
DOCKSTRING_BINARIZE = False

DOCKSTRING_DATASET_HPARAMS = {
    "ESR2": {
        "mean": -6.73,
        "noise_scale": 0.261**0.5,
        "signal_scale": 0.706,
    },
    "F2": {
        "mean": -6.14,
        "noise_scale": 0.0899**0.5,
        "signal_scale": 0.356,
    },
    "KIT": {
        "mean": -6.39,
        "noise_scale": 0.112**0.5,
        "signal_scale": 0.679,
    },
    "PARP1": {
        "mean": -6.95,
        "noise_scale": 0.0238**0.5,
        "signal_scale": 0.56,
    },
    "PGR": {
        "mean": -7.08,
        "noise_scale": 0.332**0.5,
        "signal_scale": 0.63,
    },
}
