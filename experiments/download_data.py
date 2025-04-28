from experiments.data_processing.datasets import OpenMLDataset, SGDMLDataset, UCIDataset

_SAVE_PATH = "data"

_ACSINCOME = {
    "class": OpenMLDataset,
    "class_kwargs": {"name": "acsincome", "data_folder_name": "acsincome"},
    "download_kwargs": {"save_path": _SAVE_PATH, "id": 43141},
}
_YOLANDA = {
    "class": OpenMLDataset,
    "class_kwargs": {"name": "yolanda", "data_folder_name": "yolanda"},
    "download_kwargs": {"save_path": _SAVE_PATH, "id": 42705},
}
_MALONALDEHYDE = {
    "class": SGDMLDataset,
    "class_kwargs": {"name": "malonaldehyde", "data_folder_name": "malonaldehyde"},
    "download_kwargs": {"save_path": _SAVE_PATH, "molecule": "md17_malonaldehyde"},
}
_BENZENE = {
    "class": SGDMLDataset,
    "class_kwargs": {"name": "benzene", "data_folder_name": "benzene"},
    "download_kwargs": {"save_path": _SAVE_PATH, "molecule": "md17_benzene2017"},
}
_3DROAD = {
    "class": UCIDataset,
    "class_kwargs": {"name": "3droad", "data_folder_name": "3droad"},
    "download_kwargs": {
        "save_path": _SAVE_PATH,
        "id": 246,
        "uci_file_name": "3d+road+network+north+jutland+denmark",
    },
}
_SONG = {
    "class": UCIDataset,
    "class_kwargs": {"name": "song", "data_folder_name": "song"},
    "download_kwargs": {
        "save_path": _SAVE_PATH,
        "id": 203,
        "uci_file_name": "yearpredictionmsd",
    },
}
_HOUSEELEC = {
    "class": UCIDataset,
    "class_kwargs": {"name": "houseelec", "data_folder_name": "houseelec"},
    "download_kwargs": {
        "save_path": _SAVE_PATH,
        "id": 235,
        "uci_file_name": "individual+household+electric+power+consumption",
    },
}

_DOWNLOAD_CONFIGS = [
    _ACSINCOME,
    _YOLANDA,
    _MALONALDEHYDE,
    _BENZENE,
    _3DROAD,
    _SONG,
    _HOUSEELEC,
]


if __name__ == "__main__":
    for config in _DOWNLOAD_CONFIGS:
        dataset = config["class"](**config["class_kwargs"])
        dataset.download(**config["download_kwargs"])
        print(f"Downloaded {dataset.name} dataset")
