from .operation_base import Operation
import numpy as np

__all__ = ["Arcsin",
           "Arccos",
           "Arctan",
           "Arccsc",
           "Arcsec",
           "Arccot"]


class Arcsin(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(1 < a or a < -1):
            raise ValueError("Invalid arcsin-domain value")
        return np.arcsin(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / np.sqrt(1 - self.a.data ** 2))


class Arccos(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(1 < a or a < -1):
            raise ValueError("Invalid arccos-domain value")
        return np.arccos(a.data)

    def backward_a(self, grad):
        return self.a.backward(-grad / np.sqrt(1 - self.a.data ** 2))


class Arctan(Operation):
    def __call__(self, a):
        self.a = a
        return np.arctan(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / (1 + self.a.data ** 2))


class Arccsc(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(-1 < a or a < 1):
            raise ValueError("Invalid arccsc-domain value")
        return np.arcsin(1 / a.data)

    def backward_a(self, grad):
        return self.a.backward(-grad / (np.abs(self.a.data) * np.sqrt(self.a.data ** 2 - 1)))


class Arcsec(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(-1 < a or a < 1):
            raise ValueError("Invalid arcsec-domain value")
        return np.arccos(1 / a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / (np.abs(self.a.data) * np.sqrt(self.a.data ** 2 - 1)))


class Arccot(Operation):
    def __call__(self, a):
        self.a = a
        return np.piecewise(a.data, [a.data == 0, a.data != 0], [np.pi / 2, lambda data: np.arctan(1 / a.data)])

    def backward_a(self, grad):
        return self.a.backward(-grad / (1 + self.a.data ** 2))
