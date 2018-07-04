from mygrad.tensor_base import Tensor
from mygrad.math.arithmetic.funcs import *
from mygrad.math.exp_log.funcs import *
from mygrad.math.trigonometric.funcs import *
from mygrad.math.hyperbolic_trig.funcs import *
from mygrad.math.sequential.funcs import *
from mygrad.math.sequential.funcs import max, min
from mygrad.math.misc.funcs import *
from mygrad.tensor_manip.array_shape.funcs import *
from mygrad.tensor_manip.transpose_like.funcs import *
from mygrad.tensor_creation.funcs import *
from mygrad.linalg.funcs import *

from mygrad.nnet.layers.utils import sliding_window_view

__version__ = "0.5"


for attr in (sum, prod, cumprod, cumsum,
             mean, std, var,
             max, min,
             swapaxes, transpose, moveaxis,
             reshape, squeeze,
             matmul):
    setattr(Tensor, attr.__name__, attr)
