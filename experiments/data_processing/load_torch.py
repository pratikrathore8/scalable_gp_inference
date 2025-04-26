from experiments.data_processing.datasets import (
    OpenMLDataset,
    SGDMLDataset,
    TaxiDataset,
    UCIDataset,
)

import torch

_LOAD_PATH = "data"


def _load_torch_generic(
    class_type,
    name: str,
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
    **load_kwargs
) -> tuple[torch.Tensor]:
    dataset = class_type(
        name=name,
        data_folder_name=name,
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
    )
    return dataset.load_torch(
        load_path=_LOAD_PATH,
        standardize=standardize,
        dtype=dtype,
        device=device,
        **load_kwargs,
    )


def _load_torch_openml(
    name: str,
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    return _load_torch_generic(
        OpenMLDataset,
        name=name,
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
    )


def _load_torch_sgdml(
    name: str,
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    return _load_torch_generic(
        SGDMLDataset,
        name=name,
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
    )


def _load_torch_uci(
    name: str,
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
    **load_kwargs
) -> tuple[torch.Tensor]:
    return _load_torch_generic(
        UCIDataset,
        name=name,
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
        **load_kwargs,
    )


def load_torch_acsincome(
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    return _load_torch_openml(
        name="acsincome",
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
    )


def load_torch_yolanda(
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    return _load_torch_openml(
        name="yolanda",
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
    )


def load_torch_malonaldehyde(
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    return _load_torch_sgdml(
        name="malonaldehyde",
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
    )


def load_torch_benzene(
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    return _load_torch_sgdml(
        name="benzene",
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
    )


def load_torch_taxi(
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    return _load_torch_generic(
        TaxiDataset,
        name="taxi",
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
    )


def load_torch_3droad(
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    load_kwargs = {"target_column": 3, "drop_columns": [0]}
    return _load_torch_uci(
        name="3droad",
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
        **load_kwargs,
    )


def load_torch_song(
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    load_kwargs = {"target_column": 0}
    return _load_torch_uci(
        name="song",
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
        **load_kwargs,
    )


def load_torch_houseelec(
    split_proportion: float,
    split_shuffle: bool,
    split_seed: int,
    standardize: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor]:
    load_kwargs = {"target_column": 2, "datetime_columns": [0, 1], "skip_header": True}
    return _load_torch_uci(
        name="houseelec",
        split_proportion=split_proportion,
        split_shuffle=split_shuffle,
        split_seed=split_seed,
        standardize=standardize,
        dtype=dtype,
        device=device,
        **load_kwargs,
    )


LOADERS = {
    "acsincome": load_torch_acsincome,
    "yolanda": load_torch_yolanda,
    "malonaldehyde": load_torch_malonaldehyde,
    "benzene": load_torch_benzene,
    "taxi": load_torch_taxi,
    "3droad": load_torch_3droad,
    "song": load_torch_song,
    "houseelec": load_torch_houseelec,
}
