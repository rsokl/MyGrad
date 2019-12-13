from mygrad.tensor_base import Tensor  # isort:skip  # avoid an import cycle

from mygrad.indexing_routines.funcs import *
from mygrad.linalg.funcs import *
from mygrad.math.arithmetic.funcs import *
from mygrad.math.consts import *
from mygrad.math.exp_log.funcs import *
from mygrad.math.hyperbolic_trig.funcs import *
from mygrad.math.misc.funcs import *
from mygrad.math.nondifferentiable import argmax, argmin
from mygrad.math.sequential.funcs import *
from mygrad.math.sequential.funcs import max, min
from mygrad.math.trigonometric.funcs import *
from mygrad.nnet.layers.utils import sliding_window_view
from mygrad.tensor_creation.funcs import *
from mygrad.tensor_manip.array_shape.funcs import *
from mygrad.tensor_manip.tiling.funcs import *
from mygrad.tensor_manip.transpose_like.funcs import *

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


for attr in (
    sum,
    prod,
    cumprod,
    cumsum,
    mean,
    std,
    var,
    max,
    min,
    argmax,
    argmin,
    swapaxes,
    transpose,
    moveaxis,
    reshape,
    squeeze,
    ravel,
    matmul,
):
    setattr(Tensor, attr.__name__, attr)
