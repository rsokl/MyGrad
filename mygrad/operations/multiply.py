from .operation_base import BroadcastableOp


class Multiply(BroadcastableOp):
    def __call__(self, a, b):
        self.a = a
        self.b = b

        out = a.data * b.data
        self.broadcast_check(a, b, out.shape)
        return out

    def backward_a(self, grad):
        dLda = grad * self.b.data  # dL/df * df/da -> dL/da
        broadcasted_grad = super(Multiply, self).backward_a(dLda)
        self.a.backward(broadcasted_grad)

    def backward_b(self, grad):
        dLdb = grad * self.a.data  # dL/df * df/da -> dL/da
        broadcasted_grad = super(Multiply, self).backward_b(dLdb)
        self.b.backward(broadcasted_grad)



