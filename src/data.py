import numpy as np
import torch
from enum import StrEnum

import math
from typing import Optional, Self


import pandas as pd
from .tensor import Tensor


class DataLoader:
    def __init__(
        self,
        data: np.ndarray | Tensor,
        batchsize: int,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
    ):
        if isinstance(data, Tensor):
            self.data: np.ndarray = data.value.copy()
        elif isinstance(data, np.ndarray):
            self.data: np.ndarray = data.copy()
        elif isinstance(data, list):
            self.data: np.ndarray = np.array(data)
        else:
            raise TypeError(f"Data type {type(data)} not supported")

        if shuffle:
            _rng: np.random.Generator = np.random.default_rng(shuffle_seed)
            _rng.shuffle(self.data)
        elif shuffle_seed is not None:
            print("Warning: shuffle_seed is only useful when shuffle is True")

        self.batchsize: int = batchsize
        self.idx = 0

    def __getitem__(self, idx) -> Tensor:
        if idx * self.batchsize >= len(self.data):
            raise IndexError(
                f"Index out of bounds, you're picking the {idx}. batch not the {idx} element of the data!"
            )
        # Note that we don't have to clamp `(idx + 1) * self.batchsize` because numpy does that for us
        return Tensor(self.data[idx * self.batchsize : (idx + 1) * self.batchsize])

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Tensor:
        if self.idx >= len(self.data):
            raise StopIteration
        batch = self.data[self.idx : self.idx + self.batchsize]
        self.idx += self.batchsize
        return Tensor(batch)

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batchsize)


class Data:
    def __init__(
        self,
        name: Optional[str] = None,
        url: Optional[str] = None,
        data_type: Optional[str] = None,
        sha256: Optional[str] = None,
    ):
        self.name: Optional[str] = name
        self.url: Optional[str] = url
        self.data_type: Optional[str] = data_type
        self.sha256: Optional[str] = sha256

    def download(self) -> None:
        if self.url is None:
            raise ValueError("No URL provided for this dataset {self.name}")

        if self.data_type is None:
            raise ValueError("No data type provided for this dataset {self.name}")

        # TODO: Download the file, check the sha256 and only then load the data.

        # TODO: Implement support for safetensors
        #       https://github.com/huggingface/safetensors
        match self.data_type:
            case "csv":
                self.set_data(pd.read_csv(self.url))
            case "npy":
                self.set_data(np.load(self.url))
            case "pkl":
                # Also see `pickle_arbitrary_code_example` in the scrapbook folder for a malicious example
                raise NotImplementedError(
                    ".pkl can execute arbitrary code, might not support them at all, see https://huggingface.co/docs/hub/en/security-pickle#"
                )
            case _:
                raise ValueError(f"{self.data_type=} not supported for download.")

    def verify_sha256(self, sha256: str) -> bool:
        raise NotImplementedError

    def set_data(self, data: np.ndarray | Tensor | pd.DataFrame) -> None:
        if isinstance(data, Tensor):
            self.data: np.ndarray = data.value
        elif isinstance(data, np.ndarray):
            self.data: np.ndarray = data
        elif isinstance(data, pd.DataFrame):
            self.data: np.ndarray = data.values
        elif isinstance(data, list):
            self.data: np.ndarray = np.array(data)
        else:
            raise TypeError(f"Data type {type(data)} not supported for set_data.")

    def __getitem__(self, idx) -> Tensor:
        return Tensor(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        if self.name is not None:
            temp: str = f"Dataset {self.name}:\n"
        else:
            temp: str = ""

        return temp + f"Shape: {self.data.shape}\n"

    def __repr__(self) -> str:
        return f"Data(data={self.data}, name={self.name or 'None'})"


class StatlearningNames(StrEnum):
    Advertising = "Advertising"
    Auto = "Auto"
    College = "College"
    Ch12Ex13 = "Ch12Ex13"
    Credit = "Credit"
    Heart = "Heart"
    Income1 = "Income1"
    Incomev = "Incomev"


class Statlearning(Data):
    def __init__(self, data: np.ndarray | Tensor, name: Optional[str] = None):
        if name not in [name.value for name in StatlearningNames]:
            raise ValueError(f"Name {name} not in Statlearning datasets")

        super().__init__(data, name)

        self.url = f"https://www.statlearning.com/s/{self.name}.csv"
        self.data_type = "csv"
