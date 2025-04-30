from scalable_gp_inference.hparam_training import GPHparams


# Parameters are taken from
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


def dockstring_hparams_to_gphparams(dataset: str) -> GPHparams:
    """
    Convert the dataset-specific hyperparameters to GPHparams.
    """
    if dataset not in DOCKSTRING_DATASET_HPARAMS:
        raise ValueError(f"Dataset {dataset} not found in hyperparameters.")

    hparams = DOCKSTRING_DATASET_HPARAMS[dataset]
    return GPHparams(
        signal_variance=hparams["signal_scale"],
        # NOTE(pratik): 1.0 is a placeholder value to avoid errors in the __post_init__
        kernel_lengthscale=1.0,
        noise_variance=hparams["noise_scale"] ** 2,
    )
