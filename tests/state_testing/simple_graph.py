import numpy as np

from mygrad.errors import InvalidBackprop


class SimpleOperation:
    def __call__(self, *input_vars):
        self.variables = input_vars
        raise NotImplementedError

    def backward_var(self, grad, index):
        raise NotImplementedError

    def backward(self, grad):
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index)

    def null_gradients(self, clear_graph=True):
        for var in self.variables:
            var.null_gradients(clear_graph=clear_graph)

    def clear_graph(self):
        for var in self.variables:
            var.clear_graph()


class Node:
    __array_priority__ = 15.0

    def __init__(self, x, *, constant=False, _creator=None):
        self.creator = _creator
        self.data = x.data if isinstance(x, Node) else np.asarray(x)
        self.grad = None
        self.constant = constant
        self._ops = []  # Operation instances that utilized self an input tensor

    def __repr__(self):
        return repr(self.data).replace("array", "Node").replace("\n", "\n ")

    @classmethod
    def _op(cls, Op, *input_vars, constant=False):
        tensor_vars = []
        for var in input_vars:
            if not isinstance(var, cls):
                var = cls(var, constant=True)
            var.grad = None  # participating in graph construction nulls tensor gradient
            tensor_vars.append(var)

        f = Op()
        op_out = f(*tensor_vars)
        is_const = constant or all(var.constant for var in tensor_vars)

        # record that a variable participated in that op
        for var in tensor_vars:
            var._ops.append(f)

        return cls(op_out, constant=is_const, _creator=f)

    def backward(self, grad=None, terminal_node=False):
        if self.constant:
            if terminal_node:
                self.clear_graph()
            return
        grad = np.asarray(grad) if grad is not None else np.asarray(1.0, dtype=float)
        if terminal_node:
            self.grad = np.asarray(grad)
        else:
            if not terminal_node and not self._ops:
                raise InvalidBackprop(
                    "Invalid Backprop: part of the computational graph containing "
                    "this tensor was cleared prior to backprop"
                )
            self.grad = np.asarray(grad if self.grad is None else self.grad + grad)

        if self.creator is not None:
            self.creator.backward(grad)

        if terminal_node:
            self.clear_graph()

    def clear_graph(self):
        self._ops.clear()
        if self.creator is not None:
            self.creator.clear_graph()
            self.creator = None


class Multiply(SimpleOperation):
    def __call__(self, a, b):
        self.variables = (a, b)
        out = a.data * b.data
        return out

    def backward_var(self, grad, index):
        a, b = self.variables
        if index == 0:  # backprop through a
            a.backward(grad * b.data)
        elif index == 1:  # backprop through b
            b.backward(grad * a.data)


def _multiply(a, b, constant=False):
    return Node._op(Multiply, a, b, constant=constant)


class Add(SimpleOperation):
    def __call__(self, a, b):
        self.variables = (a, b)
        out = a.data + b.data
        return out

    def backward_var(self, grad, index):
        self.variables[index].backward(np.copy(grad))


def _add(a, b, constant=False):
    return Node._op(Add, a, b, constant=constant)
