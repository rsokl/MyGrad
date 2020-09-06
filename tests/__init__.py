from numpy import ndarray

import mygrad as mg


def as_numpy(x: mg.Tensor) -> ndarray:
    """Utility for leveraging mygrad ops without incurring locked memory"""
    x.clear_graph()
    return mg.asarray(x)
