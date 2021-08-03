import numpy as np
import typing
from mygrad.tensor_base import tensor
from mygrad.tensor_base import Tensor

def save(filename : str, arr : Tensor) -> None:
    """Saves a tensor and its gradient information.

    This docstring was adapted from that of numpy.save()

    Parameters
    ----------
    file_name : str
        The desired name of the file that will hold the tensor data. Note that the file will be saved as a .npy

    arr : Tensor
        The tensor that is to be saved.

    Returns
    -------
    None
    """
    np.save(filename, arr.data)

    if arr.grad:
        gradient_path = f'{filename}_gradient'

        np.save(gradient_path, arr.grad)
    
def load(tensor_filename : str, *, gradient_filename : file or str) -> Tensor:
    """Loads a saved Tensor and its gradient information (if applicable).

    This docstring was adapted form that of numpy.load()

    Parameters
    ----------
    tensor_filename : file or str
        The name of the file that holds the tensor data to load.

    gradient_filename : file or str
        The name of the file that holds the tensor's gradient data to load.
    
    Returns
    -------
    A tensor with the desired gradient data.
    """
    loaded_tensor = tensor(np.load(tensor_filename))
    loaded_tensor.backward(np.load(gradient_filename))

    return loaded_tensor


# # if it is a tensor, save as tensor, if it is a numpy array save as numpy array
# # if it is neither, save as tensor
