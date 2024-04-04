import numpy as np
import torch

from typing import Optional, Self

from src.tensor import Tensor
from src.data import DataLoader

SEED = 0x42


def test_dataloader_magic_functions():
    _rng = np.random.default_rng(SEED)
    data: np.ndarray[np.float32] = _rng.normal(3, 2, size=(16, 6)).astype(np.float32)
    batchsize = 4

    dl_not_shuffled = DataLoader(data, batchsize, shuffle=False)

    assert len(dl_not_shuffled) == 4

    for idx, batch in enumerate(dl_not_shuffled):
        assert isinstance(batch, Tensor)
        assert np.allclose(batch.value, data[idx * batchsize : (idx + 1) * batchsize])


def test_dataloader_incomplete_batch():
    _rng = np.random.default_rng(SEED)
    data: np.ndarray[np.float32] = _rng.normal(3, 2, size=(18, 6)).astype(np.float32)
    batchsize = 4

    dl_not_shuffled = DataLoader(data, batchsize, shuffle=False)

    assert len(dl_not_shuffled) == 5

    for idx, batch in enumerate(dl_not_shuffled):
        assert batch.shape == (batchsize if idx < len(dl_not_shuffled) - 1 else 2, 6)


def test_dataloader_oversized_batch():
    _rng = np.random.default_rng(SEED)
    data: np.ndarray[np.float32] = _rng.normal(3, 2, size=(18, 6)).astype(np.float32)
    batchsize = 32

    dl_not_shuffled = DataLoader(data, batchsize, shuffle=False)

    assert len(dl_not_shuffled) == 1
    assert np.allclose(next(dl_not_shuffled).value, data)


def test_dataloader_shuffle():
    _rng = np.random.default_rng(SEED)
    data: np.ndarray[np.float32] = _rng.normal(3, 2, size=(18, 6)).astype(np.float32)
    batchsize = 4

    dl_not_shuffled = DataLoader(data, batchsize, shuffle=False)
    dl_shuffled = DataLoader(data, batchsize, shuffle=True, shuffle_seed=SEED + 1)

    assert len(dl_not_shuffled) == len(dl_shuffled)
    assert not np.allclose(dl_not_shuffled.data, dl_shuffled.data)
