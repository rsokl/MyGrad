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



class OldRecurrentUnit(Operation):
    def __init__(self, U, W, V, bp_lim):
        self.U = U
        self.W = W
        self.V = V
        self.bp_lim = bp_lim

        self._input_seq = None
        self._hidden_seq = []

        self.bp_cnt = 0

    def __call__(self, seq, s0=None):

        if self._input_seq is None:
            self._input_seq = seq
        else:
            self._input_seq = np.vstack((self._input_seq, seq))

        out = np.zeros((seq.shape[0] + 1, seq.shape[1], self.U.shape[-1]))

        if self._hidden_seq:
            out[0] = self._hidden_seq[-1].data
        elif s0 is not None:
            out[0] = s0 if isinstance(s0, Tensor) else s0

        np.dot(seq, self.U.data, out=out[1:])
        for n, s_prev in enumerate(out[:-1]):
            out[n + 1] += s_prev.dot(self.W.data)
            out[n + 1] = np.tanh(out[n + 1])

        if not self._hidden_seq:
            self._hidden_seq = [Tensor(out[0], _creator=self, _seq_index=0)]

        self._hidden_seq += [Tensor(s, _creator=self, _seq_index=(n + len(self._hidden_seq)))
                             for n, s in enumerate(out[1:])]
        return self._hidden_seq

    def backward(self, grad, seq_index=None):
        """ o = UX_t + WS_{t-1}
            S_{t} = tanh(o)"""

        self.bp_cnt += 1

        s = self._hidden_seq[seq_index]
        s_prev = self._hidden_seq[seq_index - 1]
        x = self._input_seq[seq_index - 1]

        dLdo = grad * (1 - s.data ** 2)
        self.U.backward(np.dot(x.T, dLdo))
        self.W.backward(np.dot(s_prev.data.T, dLdo))

        s.grad = None
        if self.bp_cnt == self.bp_lim or seq_index == 1:
            self.bp_cnt = 0
        else:
            s_prev.grad = None
            s_prev.backward(np.dot(dLdo, self.W.data.T))








