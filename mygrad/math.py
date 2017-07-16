from .tensor_base import Tensor
from .operations import *

def add(a, b):
    return Tensor._op(Add, a, b)

def multiply(a, b):
    return Tensor._op(Multiply, a, b)

def multiply_sequence(*variables):
    return Tensor._op(MultiplySequence, *variables)

def divide(a, b):
    return Tensor._op(Divide, a, b)