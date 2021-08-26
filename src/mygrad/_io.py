from pathlib import Path
from typing import BinaryIO, Union

import numpy as np

import mygrad.tensor_base as tb

_FileLike = Union[str, Path, BinaryIO]


def save(file: _FileLike, tensor: tb.Tensor) -> None:
    """Saves a tensor and its gradient information.

    This docstring was adapted from that of numpy.save()

    Parameters
    ----------
    file : str | Path | BinaryIO
        The file or file-path that where the tensor data and its gradient
        will be saved. Note that the file will be saved as a .npz file.

    tensor : Tensor
        The tensor to be saved. If it has an associated gradient, that will
        be saved as well.

    Notes
    -----
    This function uses ``numpy.savez(file, data=tensor.data, grad=tensor.grad)``
    to save the tensor's data and its gradient. No ``grad`` field is included
    if the tensor does not have a gradient.

    See Also
    --------
    mygrad.load

    Examples
    --------
    >>> import mygrad as mg
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = mg.arange(10.0)
    >>> mg.save(outfile, x)
    >>> _ = outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> mg.load(outfile)
    Tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    An example of saving a tensor that has an associated gradient.

    >>> (x * x).backward()
    >>> x.grad
    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])
    >>> outfile = TemporaryFile()
    >>> x = mg.arange(10.0)
    >>> mg.save(outfile, x)
    >>> _ = outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> loaded = mg.load(outfile)
    >>> loaded
    Tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> loaded.grad
    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])
    """
    if not isinstance(tensor, tb.Tensor):
        raise TypeError(
            f"mygrad.save requires a Tensor-type object, got type {type(tensor)}"
        )

    if tensor.grad is not None:
        np.savez(file, data=tensor.data, grad=tensor.grad)
    else:
        np.savez(file, data=tensor.data)


def load(file: _FileLike) -> tb.Tensor:
    """Loads a saved Tensor and its gradient information (if applicable).

    This docstring was adapted from that of numpy.load()

    Parameters
    ----------
    file : str | Path | BinaryIO
        The name of the file that holds the tensor data to load.

    Returns
    -------
    loaded : Tensor
        The loaded tensor (whose gradient will be loaded if it was saved).

    See Also
    --------
    mygrad.save

    Examples
    --------
    >>> import mygrad as mg
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = mg.arange(10.0)
    >>> mg.save(outfile, x)
    >>> _ = outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> mg.load(outfile)
    Tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    An example of saving a tensor that has an associated gradient.

    >>> (x * x).backward()
    >>> x.grad
    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])
    >>> outfile = TemporaryFile()
    >>> x = mg.arange(10.0)
    >>> mg.save(outfile, x)
    >>> _ = outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> loaded = mg.load(outfile)
    >>> loaded
    Tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> loaded.grad
    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])
    """
    loaded = np.load(file)

    loaded_tensor = tb.tensor(loaded["data"])

    if "grad" in loaded:
        loaded_tensor.backward(loaded["grad"])

    return loaded_tensor
