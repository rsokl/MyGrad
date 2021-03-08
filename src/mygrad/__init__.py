from mygrad.tensor_base import (  # isort:skip  # avoid an import cycle
    asarray,
    astensor,
    tensor,
    Tensor,
)
from mygrad._utils.graph_tracking import no_autodiff
from mygrad._utils.lock_management import (
    mem_guard_active,
    mem_guard_off,
    mem_guard_on,
    turn_memory_guarding_off,
    turn_memory_guarding_on,
)
from mygrad.indexing_routines.funcs import *
from mygrad.linalg.funcs import *
from mygrad.math.arithmetic.funcs import *
from mygrad.math.consts import *
from mygrad.math.exp_log.funcs import *
from mygrad.math.hyperbolic_trig.funcs import *
from mygrad.math.misc.funcs import *
from mygrad.math.nondifferentiable import any, argmax, argmin
from mygrad.math.sequential.funcs import *
from mygrad.math.sequential.funcs import max, min
from mygrad.math.trigonometric.funcs import *
from mygrad.nnet.layers.utils import sliding_window_view
from mygrad.no_grad_funcs import *
from mygrad.tensor_creation.funcs import *
from mygrad.tensor_manip.array_shape.funcs import *
from mygrad.tensor_manip.tiling.funcs import *
from mygrad.tensor_manip.transpose_like.funcs import *
from mygrad.ufuncs._ufunc_creators import ufunc

from . import random
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


setattr(Tensor, "clip", clip)
