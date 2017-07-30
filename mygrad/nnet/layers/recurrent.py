from ...operations.operation_base import Operation
from ...tensor_base import Tensor
import numpy as np


class RecurrentUnit(Operation):
    def __init__(self, U, W, V, bp_lim):
        self.U = U
        self.W = W
        self.V = V
        self.bp_lim = bp_lim

        self._input_seq = []
        self._hidden_seq = []

        self.bp_cnt = 0

    def __call__(self, s_old, seq=None):
        if not self._hidden_seq:
            s_old._seq_index = len(self._hidden_seq)
            self._hidden_seq.append(s_old)

        assert s_old._seq_index == len(self._hidden_seq) - 1
        self._input_seq.append(seq)

        out = seq.dot(self.U.data)
        for n, s_prev in enumerate(out[:-1]):
            out[n + 1] += s_old.data.dot(self.W.data)

        np.tanh(out, out=out)

        s = Tensor(out, _creator=self)
        s._seq_index = len(self._hidden_seq)
        self._hidden_seq.append(s)
        return s

    # def __call__(self, s_old, seq=None):
    #     if not self._hidden_seq:
    #         s_old._seq_index = len(self._hidden_seq)
    #         self._hidden_seq.append(s_old)
    #
    #     assert s_old._seq_index == len(self._hidden_seq) - 1
    #     self._input_seq.append(seq)
    #
    #     out = seq.dot(self.U.data)
    #     out += s_old.data.dot(self.W.data)
    #     np.tanh(out, out=out)
    #
    #     s = Tensor(out, _creator=self)
    #     s._seq_index = len(self._hidden_seq)
    #     self._hidden_seq.append(s)
    #     return s

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


def recurrent(s, x):
    return Tensor._op(RecurrentUnit, s, op_kwargs={"seq": x})









