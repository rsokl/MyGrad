from pathlib import Path
from typing import Union

import numpy as np

import mygrad.tensor_base as tb


def save(file: Union[str, Path], tensor: tb.Tensor) -> None:
    """Saves a tensor and its gradient information.

    This docstring was adapted from that of numpy.save()

    Parameters
    ----------
    file : str
        The desired name of the file that will hold the tensor data. Note that the file will be saved as a .npz

    tensor : Tensor
        The tensor that is to be saved, along with its gradient information.

    Returns
    -------
    None
    """
    if not isinstance(tensor, tb.Tensor):
        raise TypeError(
            f"mygrad.save requires a Tensor-type object, got type {type(tensor)}"
        )

    if tensor.grad is not None:
        np.savez(file, data=tensor.data, grad=tensor.grad)
    else:
        np.savez(file, data=tensor.data)


def load(file: Union[str, Path]) -> tb.Tensor:
    """Loads a saved Tensor and its gradient information (if applicable).

    This docstring was adapted from that of numpy.load()

    Parameters
    ----------
    file : str
        The name of the file that holds the tensor data to load.

    Returns
    -------
    A tensor with the desired gradient data.
    """
    loaded = np.load(file)

    loaded_tensor = tb.tensor(loaded["data"])

    if "grad" in loaded:
        loaded_tensor.backward(loaded["grad"])

    return loaded_tensor
