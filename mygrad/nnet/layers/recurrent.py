from ...operations.operation_base import Operation
from ...tensor_base import Tensor
import numpy as np

from numba import njit


@njit
def _dot_tanh_accum(x, W):
    for n in range(len(x) - 1):
        x[n + 1] += np.dot(x[n], W)
        x[n + 1] = np.tanh(x[n + 1])


class RecurrentUnit(Operation):
    def __init__(self, U, W, V, bp_lim):
        self.U = U
        self.W = W
        self.V = V
        self.bp_lim = bp_lim

        self._input_seq = None
        self._hidden_seq = []

        self.bp_cnt = 0

    def __call__(self, seq, s0=None):

        self._input_seq = seq if self._input_seq is None else np.vstack((self._input_seq, seq))

        out = np.zeros((seq.shape[0] + 1, seq.shape[1], self.U.shape[-1]))

        if self._hidden_seq:
            out[0] = self._hidden_seq[-1].data
        elif s0 is not None:
            out[0] = s0.data if isinstance(s0, Tensor) else s0

        np.dot(seq, self.U.data, out=out[1:])
        _dot_tanh_accum(out, self.W.data)

        if not self._hidden_seq:
            self._hidden_seq = Tensor(out, _creator=self)
        else:
            new_dat = np.vstack((self._hidden_seq.data, out[1:]))
            self._hidden_seq = Tensor(new_dat, _creator=self)

        return self._hidden_seq


    def backward(self, grad, seq_index=None):
        s = self._hidden_seq

        dsdf = (1 - s.data ** 2)
        grad = grad * dsdf
        old_grad = np.zeros_like(grad)
        for i in range(min(s.shape[0] - 1, self.bp_lim)):
            dt = grad[2:len(grad) - i] - old_grad[2:len(grad) - i]
            old_grad = np.copy(grad)
            grad[1:len(grad) - (i + 1)] += dsdf[1:len(grad) - (i + 1)] * np.dot(dt, self.W.data.T)

        self.U.backward(np.einsum("ijk, ijl -> kl", self._input_seq, grad[1:]))
        self.W.backward(np.einsum("ijk, ijl -> kl", s.data[:-1], grad[1:]))


@njit
def _gru_layer(s, z, r, h, Wz, Wr, Wh):
    for n in range(len(s) - 1):
        z[n] += np.dot(s[n], Wz)
        z[n] = 1 / (1 + np.exp(-z[n]))

        r[n] += np.dot(s[n], Wr)
        r[n] = 1 / (1 + np.exp(-r[n]))

        h[n] += np.dot(r[n] * s[n], Wh)
        h[n] = np.tanh(h[n])

        s[n + 1] = (1 - z[n]) * h[n] + z[n] * s[n]


class GRUnit(Operation):
    def __init__(self, Uz, Wz, Ur, Wr, Uh, Wh, V, bp_lim):
        self.Uz = Uz
        self.Wz = Wz

        self.Ur = Ur
        self.Wr = Wr

        self.Uh = Uh
        self.Wh = Wh

        self.V = V

        self.bp_lim = bp_lim

        self._input_seq = None
        self._hidden_seq = []

        self._z = []
        self._r = []
        self._h = []

        self.bp_cnt = 0


    def __call__(self, seq, s0=None):
        # fix W's and U's references
        self._input_seq = seq if self._input_seq is None else np.vstack((self._input_seq, seq))

        z = np.zeros((seq.shape[0], seq.shape[1], self.Uz.shape[-1]))
        r = np.zeros((seq.shape[0], seq.shape[1], self.Ur.shape[-1]))
        h = np.zeros((seq.shape[0], seq.shape[1], self.Uh.shape[-1]))
        out = np.zeros((seq.shape[0] + 1, seq.shape[1], self.Uz.shape[-1]))

        if self._hidden_seq:
            out[0] = self._hidden_seq[-1].data
        elif s0 is not None:
            out[0] = s0.data if isinstance(s0, Tensor) else s0

        np.dot(seq, self.Uz.data, out=z)
        np.dot(seq, self.Ur.data, out=r)
        np.dot(seq, self.Uh.data, out=h)

        _gru_layer(out, z, r, h, self.Wz.data, self.Wr.data, self.Wh.data)


        if not self._hidden_seq:
            self._hidden_seq = Tensor(out, _creator=self)
        else:
            new_dat = np.vstack((self._hidden_seq.data, out))
            self._hidden_seq = Tensor(new_dat, _creator=self)

        if not self._z:
            self._z = Tensor(z, _creator=self)
        else:
            new_dat = np.vstack((self._z.data, z))
            self._z = Tensor(new_dat, _creator=self)

        if not self._r:
            self._r = Tensor(r, _creator=self)
        else:
            new_dat = np.vstack((self._r.data, r))
            self._r = Tensor(new_dat, _creator=self)

        if not self._h:
            self._h = Tensor(h, _creator=self)
        else:
            new_dat = np.vstack((self._h.data, h))
            self._h = Tensor(new_dat, _creator=self)

        return self._hidden_seq


    def backward(self, grad, seq_index=None):
        s = self._hidden_seq.data[:-1]
        z = self._z.data
        r = self._r.data
        h = self._h.data

        # dsds = z + dsdz * (z * (1 - z)) + dsdh * (1 - s ** 2) + dsdr * (r * (1 - r))

        pdsds = z
        dsdz = -h + s
        dsdh = 1 - z
        pdzds = np.dot(z * dsdh, self.Wz.data.T)
        pdh = (1 - h ** 2) * (np.dot(s * r, self.Wh.data.T))
        pdhds = pdh * r
        pdhdr = pdh * s
        pdrds = np.dot(r * (1 - r), self.Wr.data.T)

        dsds = pdsds + dsdz * pdzds + dsdh * (pdhds + pdhdr * pdrds)
        dsdr = dsdh * pdhdr

        # remove mult, passes for Wz.grad/fails for Uz.grad
        # readd, passes for all
        grad = grad[1:]

        old_grad = np.zeros_like(grad)

        for i in range(min(s.shape[0] - 1, self.bp_lim)):
            dt = grad[1:len(grad) - i] - old_grad[1:len(grad) - i]
            old_grad = np.copy(grad)
            grad[:len(grad) - (i + 1)] += dsds[:len(grad) - (i + 1)] * dt

        zgrad = grad * dsdz
        rgrad = grad * dsdr
        hgrad = grad * dsdh

        # self._z.grad = zgrad
        # self._r.grad = rgrad
        # self._h.grad = hgrad

        self.Uz.backward(np.einsum("ijk, ijl -> kl", self._input_seq, zgrad))
        self.Wz.backward(np.einsum("ijk, ijl -> kl", s, zgrad))

        self.Ur.backward(np.einsum("ijk, ijl -> kl", self._input_seq, rgrad))
        self.Wr.backward(np.einsum("ijk, ijl -> kl", s, rgrad))

        self.Uh.backward(np.einsum("ijk, ijl -> kl", self._input_seq, hgrad))
        self.Wh.backward(np.einsum("ijk, ijl -> kl", (s * r), hgrad))

        """
        dsdf = (1 - s.data ** 2)
        grad = grad * dsdf
        old_grad = np.zeros_like(grad)
        for i in range(min(s.shape[0] - 1, self.bp_lim)):
            dt = grad[2:len(grad) - i] - old_grad[2:len(grad) - i]
            old_grad = np.copy(grad)
            grad[1:len(grad) - (i + 1)] += dsdf[1:len(grad) - (i + 1)] * np.dot(dt, self.W.data.T)

        self.U.backward(np.einsum("ijk, ijl -> kl", self._input_seq, grad[1:]))
        self.W.backward(np.einsum("ijk, ijl -> kl", s.data[:-1], grad[1:]))"""
