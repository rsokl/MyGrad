from .operation_base import Operation
import numpy as np

__all__ = ["Arcsinh",
           "Arccosh",
           "Arctanh",
           "Arccsch",
           "Arcsech",
           "Arccoth"]


class Arcsinh(Operation):
    def __call__(self, a):
        self.a = a
        return np.arcsinh(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / np.sqrt(1 + self.a.data ** 2))


class Arccosh(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a < 1):
            raise ValueError("Invalid arccosh-domain value")
        return np.arccosh(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / np.sqrt(self.a.data ** 2 - 1))


class Arctanh(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(1 <= a or a <= -1):
            raise ValueError("Invalid arctanh-domain value")
        return np.arctanh(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / (1 - self.a.data ** 2))


class Arccsch(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a == 0):
            raise ValueError("Invalid arccsch-domain value")
        return np.arcsinh(1 / a.data)

    def backward_a(self, grad):
        return self.a.backward(-grad / (np.abs(self.a.data) * np.sqrt(1 + self.a.data ** 2)))


class Arcsech(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(1 < a or a <= 0):
            raise ValueError("Invalid arcsech-domain value")
        return np.arccosh(1 / a.data)

    def backward_a(self, grad):
        return self.a.backward(-grad / (self.a.data * np.sqrt(1 - self.a.data ** 2)))


class Arccoth(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(-1 <= a or a <= 1):
            raise ValueError("Invalid arccoth-domain value")
        return np.arctanh(1 / a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / (1 - self.a.data ** 2))
