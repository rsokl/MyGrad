from .operation_base import Operation
import numpy as np

__all__ = ["Sin",
           "Cos",
           "Tan",
           "Csc",
           "Sec",
           "Cot"]


class Sin(Operation):
    def __call__(self, a):
        self.a = a
        return np.sin(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * np.cos(self.a.data))


class Cos(Operation):
    def __call__(self, a):
        self.a = a
        return np.cos(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * -np.sin(self.a.data))


class Tan(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a.data % (np.pi / 2) == 0) and np.any(a.data % np.pi != 0):
            raise ValueError("Invalid tan-domain value")
        return np.tan(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad / np.cos(self.a.data)**2)


class Csc(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a.data % np.pi == 0):
            raise ValueError("Invalid csc-domain value")
        return 1 / np.sin(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * -np.cos(self.a.data) / np.sin(self.a.data)**2)


class Sec(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a.data % (np.pi / 2) == 0) and np.any(a % np.pi != 0):
            raise ValueError("Invalid sec-domain value")
        return 1 / np.cos(a.data)

    def backward_a(self, grad):
        return self.a.backward(grad * np.sin(self.a.data) / np.cos(self.a.data)**2)


class Cot(Operation):
    def __call__(self, a):
        self.a = a
        if np.any(a.data % np.pi == 0):
            raise ValueError("Invalid cot-domain value")
        return 1 / np.tan(a.data)

    def backward_a(self, grad):
        return self.a.backward(-grad / np.sin(self.a.data)**2)
