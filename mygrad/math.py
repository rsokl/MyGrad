from .tensor_base import Tensor
from .operations import *

def add(*variables):
    return Tensor._op(AddSequence, *variables)

def multiply(*variables):
    return Tensor._op(MultiplySequence, *variables)
