from mygrad.tensor_creation.funcs import identity

from .constant import constant
from .dirac import dirac
from .glorot_normal import glorot_normal
from .glorot_uniform import glorot_uniform
from .he_normal import he_normal
from .he_uniform import he_uniform
from .normal import normal
from .uniform import uniform

__all__ = [
    "constant",
    "dirac",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "identity",
    "normal",
    "uniform",
]
