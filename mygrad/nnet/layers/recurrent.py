from ...operations.operation_base import Operation
from ...tensor_base import Tensor
from numbers import Integral
import numpy as np

from numba import njit


@njit
def _dot_tanh_accum(x, W):
    for n in range(len(x) - 1):
        x[n + 1] += np.dot(x[n], W)
        x[n + 1] = np.tanh(x[n + 1])


class RecurrentUnit(Operation):
    """ Defines a basic recurrent unit for a RNN.

        This unit operates on a sequence of data {X_j | (0 <= j <= T - 1)}, producing
        a sequence of "hidden descriptors": {S_i | (0 <= i <= T}}, via the trainable parameters
        U and W

                                S_{t} = tanh(U X_{t-1} + W S_{t-1})

        For a language model, S_{t} is traditionally mapped to a prediction of X_t via: softmax(V S_t),
        where V is yet another trainable parameter (not built into the recurrent unit)."""

    def __call__(self, X, U, W, s0=None, bp_lim=None):
        """ Performs a forward pass of sequential data through a simple RNN layer, returning
            the 'hidden-descriptors' arrived at by utilizing the trainable parameters U and V:

                                S_{t} = tanh(U X_{t-1} + W S_{t-1})

            Parameters
            ----------
            X : mygrad.Tensor, shape=(T, N, C)
               The sequential data to be passed forward.

            U : mygrad.Tensor, shape=(D, C)
               The weights used to map sequential data to its hidden-descriptor representation

            W : mygrad.Tensor, shape=(D, D)
                The weights used to map a hidden-descriptor to a hidden-descriptor.

            s0 : Optional[mygrad.Tensor, numpy.ndarray], shape=(N, D)
                The 'seed' hidden descriptors to feed into the RNN. If None, a Tensor
                of zeros of shape (N, D) is created.

            bp_lim : Optional[int]
                The (non-zero) limit of the number of back propagations through time are
                performed


            Returns
            -------
            mygrad.Tensor
                The sequence of 'hidden-descriptors' produced by the forward pass of the RNN."""
        if bp_lim is not None:
            assert isinstance(bp_lim, Integral) and 0 < bp_lim <= len(X)
        self.bp_lim = bp_lim if bp_lim is not None else len(X)

        self.X = X
        self.U = U
        self.W = W
        self._hidden_seq = []

        seq = self.X.data
        out = np.zeros((seq.shape[0] + 1, seq.shape[1], self.U.shape[-1]))

        if s0 is not None:
            out[0] = s0.data if isinstance(s0, Tensor) else s0

        np.dot(seq, self.U.data, out=out[1:])
        _dot_tanh_accum(out, self.W.data)

        self._hidden_seq = Tensor(out, _creator=self)

        return self._hidden_seq.data

    def backward(self, grad):
        """ Performs back propagation through time (with optional truncation), using the
            following notation:
                s_t = tanh(f_t)
                f_t = U x_{t-1} + W s_{t-1}
        """
        if self.U.constant and self.W.constant and self.X.constant:
            return None

        s = self._hidden_seq

        dst_dft = (1 - s.data ** 2)  # ds_{t} / d_f{t}
        dLt_dst = np.copy(grad)  # dL_{t} / ds_{t}
        old_dst = np.zeros_like(grad)

        for i in range(self.bp_lim):
            # dL_{n} / ds_{t+1} -> dL_{n} / df_{t+1}  | ( n > t )
            index = slice(2, len(grad) - i)
            dLn_ft1 = dst_dft[index] * (dLt_dst[index] - old_dst[index])
            old_dst = np.copy(dLt_dst)
            dLt_dst[1:len(grad) - (i + 1)] += np.dot(dLn_ft1, self.W.data.T)  # dL_{t} / ds_{t} + ... + dL_{n} / ds_{t}

        self._hidden_seq.grad = dLt_dst  # element t: dL_{t} / ds_{t} + ... + dL_{T_lim} / ds_{t}

        dLt_dft = dLt_dst[1:] * dst_dft[1:]  # dL_{t} / df_{t} + ... + dL_{T_lim} / df_{t}

        if not self.U.constant:
            self.U.backward(np.einsum("ijk, ijl -> kl", self.X.data, dLt_dft))  # dL_{1} / dU + ... + dL_{T} / dU
        if not self.W.constant:
            self.W.backward(np.einsum("ijk, ijl -> kl", s.data[:-1], dLt_dft))  # dL_{1} / dW + ... + dL_{T} / dW
        if not self.X.constant:
            self.X.backward(np.dot(dLt_dft, self.U.data.T))  # dL_{1} / dX + ... + dL_{T} / dX

    def null_gradients(self):
        """ Back-propagates `None` to the gradients of the operation's input Tensors."""
        for x in [self.X, self.U, self.W]:
            x.null_gradients()


def simple_RNN(X, U, W, s0=None, bp_lim=None):
    """ Performs a forward pass of sequential data through a simple RNN layer, returning
        the 'hidden-descriptors' arrived at by utilizing the trainable parameters U and V:

                            S_{t} = tanh(U X_{t-1} + W S_{t-1})

        Parameters
        ----------
        X : mygrad.Tensor, shape=(T, N, C)
           The sequential data to be passed forward.

        U : mygrad.Tensor, shape=(D, C)
           The weights used to map sequential data to its hidden-descriptor representation

        W : mygrad.Tensor, shape=(D, D)
            The weights used to map a hidden-descriptor to a hidden-descriptor.

        s0 : Optional[mygrad.Tensor, numpy.ndarray], shape=(N, D)
            The 'seed' hidden descriptors to feed into the RNN. If None, a Tensor
            of zeros of shape (N, D) is created.

        bp_lim : Optional[int]
            The (non-zero) limit of the number of back propagations through time are
            performed

        backprop_s : bool, (default=False)
            If False, backpropagation will not be carried through the hidden-descriptors of
            this RNN. Backpropagation through X, U, and W will still occur, granted that
            these are non-constant Tensors.

        Returns
        -------
        mygrad.Tensor
            The sequence of 'hidden-descriptors' produced by the forward pass of the RNN.

        Notes
        -----
        T : Sequence length
        N : Batch size
        C : Length of single datum
        D : Length of 'hidden' descriptor"""
    s = Tensor._op(RecurrentUnit, X, U, W, op_kwargs=dict(s0=s0, bp_lim=bp_lim))
    s.creator._hidden_seq = s
    return s
