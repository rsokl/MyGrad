from ..operations.operation_base import Operation


class Reshape(Operation):
    def __call__(self, a, shape):
        self.a = a
        return a.data.reshape(shape)

    def backward_a(self, grad):
        self.a.backward(grad.reshape(*self.a.shape))
