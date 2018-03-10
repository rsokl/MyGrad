from .operation_base import Operation
import numpy as np

__all__ = ["Sinh",
           "Cosh",
           "Tanh",
           "Csch",
           "Sech",
           "Coth"]

class Sinh(Operation):
    def __call__(self, a):
        self.a = a
        return np.sinh(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * np.cosh(self.a.data))


class Cosh(Operation):
    def __call__(self, a):
        self.a = a
        return np.cosh(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * np.sinh(self.a.data))


class Tanh(Operation):
    def __call__(self, a):
        self.a = a
        return np.tanh(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * (1 - np.tanh(self.a.data) ** 2))


class Csch(Operation):
    def __call__(self, a):
        self.a = a
        return 1 / np.sinh(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * -np.cosh(self.a.data) / np.sinh(self.a.data)**2)


class Sech(Operation):
    def __call__(self, a):
        self.a = a
        return 1 / np.cosh(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * -np.sinh(self.a.data) / np.cosh(self.a.data)**2)


class Coth(Operation):
    def __call__(self, a):
        self.a = a
        return 1 / np.tanh(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * -1 / np.sinh(self.a.data)**2)
