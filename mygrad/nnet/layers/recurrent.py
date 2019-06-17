from numbers import Integral

import numpy as np

from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor

try:
    from numba import njit
except ImportError:
    raise ImportError(
        "The package `numba` must be installed in order to access the simple-rnn."
    )


@njit
def dot(a, b):
    """
    Calculates the dot product between 2 arrays
    of shapes (W,X,Y) and (Y,Z), respectively
    """
    out = np.zeros((a.shape[0], a.shape[1], b.shape[-1]))
    for i in range(len(a)):
        out[i] = np.dot(a[i], b)
    return out


@njit
def _dot_tanh_accum(x, W):
    for n in range(len(x) - 1):
        x[n + 1] += np.dot(x[n], W)
        x[n + 1] = np.tanh(x[n + 1])


@njit
def _rnn_bptt(X, dLt_dst, dst_dft, W, bp_lim, old_dst=None):
    W = W.T
    if bp_lim < len(X) - 1:
        old_dst = np.zeros_like(dLt_dst)

    for i in range(bp_lim):
        if bp_lim < len(X) - 1:
            source_index = slice(2, len(dLt_dst) - i)
            target_index = slice(1, len(dLt_dst) - (i + 1))

            # dL_{n} / ds_{t+1} -> dL_{n} / df_{t+1}  | ( n > t )
            dLn_ft1 = dst_dft[source_index] * (
                dLt_dst[source_index] - old_dst[source_index]
            )
            old_dst = np.copy(dLt_dst)

        else:  # no backprop truncation
            source_index = slice(len(dLt_dst) - (i + 1), len(dLt_dst) - i)
            target_index = slice(len(dLt_dst) - (i + 2), len(dLt_dst) - (i + 1))
            dLn_ft1 = dst_dft[source_index] * dLt_dst[source_index]

        dLt_dst[target_index] += dot(dLn_ft1, W)


def _backprop(var, grad):
    if not var.constant:
        if var.grad is None:
            var.grad = np.asarray(grad)
        else:
            var.grad += grad


class RecurrentUnit(Operation):
    """ Defines a basic recurrent unit for a RNN.

        This unit operates on a sequence of data {X_j | (0 <= j <= T - 1)}, producing
        a sequence of "hidden descriptors": {S_i | (0 <= i <= T}}, via the trainable parameters
        U and W

                                S_{t} = tanh(U X_{t-1} + W S_{t-1})

        For a language model, S_{t} is traditionally mapped to a prediction of X_t via: softmax(V S_t),
        where V is yet another trainable parameter (not built into the recurrent unit)."""

    scalar_only = True

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
            assert isinstance(bp_lim, Integral) and 0 <= bp_lim < len(X)
        self.bp_lim = bp_lim if bp_lim is not None else len(X) - 1

        self.X = X
        self.U = U
        self.W = W
        self.variables = (self.X, self.U, self.W)
        self._hidden_seq = []

        seq = self.X.data
        out = np.zeros((seq.shape[0] + 1, seq.shape[1], self.U.shape[-1]))

        if s0 is not None:
            out[0] = s0.data if isinstance(s0, Tensor) else s0

        out[1:] = dot(seq, self.U.data)
        _dot_tanh_accum(out, self.W.data)

        self._hidden_seq = Tensor(out, _creator=self)

        return self._hidden_seq.data

    def backward(self, grad, *, graph, **kwargs):
        """ Performs back propagation through time (with optional truncation), using the
            following notation:

                f_t = U x_{t-1} + W s_{t-1}
                s_t = tanh(f_t)
        """
        if self.U.constant and self.W.constant and self.X.constant:
            return None

        s = self._hidden_seq

        dst_dft = 1 - s.data ** 2  # ds_{t} / d_f{t}
        dLt_dst = np.copy(grad)  # dL_{t} / ds_{t}

        _rnn_bptt(self.X.data, dLt_dst, dst_dft, self.W.data, self.bp_lim)

        self._hidden_seq.grad = (
            dLt_dst
        )  # element t: dL_{t} / ds_{t} + ... + dL_{T_lim} / ds_{t}

        dLt_dft = (
            dLt_dst[1:] * dst_dft[1:]
        )  # dL_{t} / df_{t} + ... + dL_{T_lim} / df_{t}

        if not self.U.constant:
            _backprop(
                self.U, np.einsum("ijk, ijl -> kl", self.X.data, dLt_dft)
            )  # dL_{1} / dU + ... + dL_{T} / dU
        if not self.W.constant:
            _backprop(
                self.W, np.einsum("ijk, ijl -> kl", s.data[:-1], dLt_dft)
            )  # dL_{1} / dW + ... + dL_{T} / dW
        if not self.X.constant:
            _backprop(
                self.X, dot(dLt_dft, self.U.data.T)
            )  # dL_{1} / dX + ... + dL_{T} / dX


def simple_RNN(X, U, W, s0=None, bp_lim=None, constant=False):
    """ Performs a forward pass of sequential data through a simple RNN layer, returning
        the 'hidden-descriptors' arrived at by utilizing the trainable parameters U and V::

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
            The (non-zero) limit of the depth of back propagation through time to be
            performed. If `None` back propagation is passed back through the entire sequence.

            E.g. `bp_lim=3` will propagate gradients only up to 3 steps backward through the
            recursive sequence.

        constant : bool, optional (default=False)
            If True, the resulting Tensor is a constant.

        Returns
        -------
        mygrad.Tensor
            The sequence of 'hidden-descriptors' produced by the forward pass of the RNN.

        Notes
        -----
        This simple RNN implements the recursive system of equations:

        .. math::
          S_{t} = tanh(U X_{t-1} + W S_{t-1})

        - :math:`T` : Sequence length
        - :math:`N` : Batch size
        - :math:`C` : Length of single datum
        - :math:`D` : Length of 'hidden' descriptor"""
    s = Tensor._op(
        RecurrentUnit, X, U, W, op_kwargs=dict(s0=s0, bp_lim=bp_lim), constant=constant
    )
    s.creator._hidden_seq = s
    return s
