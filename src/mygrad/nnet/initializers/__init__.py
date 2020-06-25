from mygrad.tensor_creation.funcs import identity
from .constant import constant
from .dirac import dirac
from .glorot_normal import glorot_normal
from .he_normal import he_normal
from .he_uniform import he_uniform
from .normal import normal
from .uniform import uniform


__all__ = [
    "constant",
    "dirac",
    "glorot_normal",
    "he_normal",
    "he_uniform",
    "identity",
    "normal",
    "uniform",
]
