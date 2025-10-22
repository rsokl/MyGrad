import weakref
from numbers import Integral

import numpy as np

from mygrad._utils import SkipGradient
from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor

try:
    from numba import njit, vectorize
except ImportError:  # pragma: no cover
    raise ImportError(
        "The package `numba` must be installed in order to access the gru."
    )


@vectorize(
    ["float32(float32)", "float64(float64)"],
    nopython=True,
)
def sig(f):  # pragma: no cover
    """
    Calculates a sigmoid function
    """
    return 1 / (1 + np.exp(-f))


@vectorize(
    ["float32(float32)", "float64(float64)"],
    nopython=True,
)
def d_sig(f):  # pragma: no cover
    """
    Calculates the derivative of a sigmoid function
    """
    return f * (1 - f)


@vectorize(
    ["float32(float32)", "float64(float64)"],
    nopython=True,
)
def d_tanh(f):  # pragma: no cover
    """
    Calculates the derivative of a tanh function
    """
    return 1 - f**2


@njit
def dot(a, b):
    """
    Calculates the dot product between 2 arrays
    of shapes (W,X,Y) and (Y,Z), respectively
    """
    return np.dot(a.reshape(-1, a.shape[-1]), b).reshape(*a.shape[:-1], b.shape[-1])


@njit
def _gru_layer(s, z, r, h, Wz, Wr, Wh):
    """Given:
        S(t=0)
        z = X(t) Uz + bz
        r = X(t) Ur + br
        h = X(t) Uh + bh

    Compute Z(t), R(t), H(t), S(t) for all 1 <= t <= T

    Parameters
    ----------
    s : numpy.ndarray, shape=(T+1, N, D)
        Modified in-place
    z : numpy.ndarray, shape=(T, N, D)
        Modified in-place
    r : numpy.ndarray, shape=(T, N, D)
        Modified in-place
    h : numpy.ndarray, shape=(T, N, D)
        Modified in-place
    Wz : numpy.ndarray, shape=(D, D)
    Wr : numpy.ndarray, shape=(D, D)
    Wh : numpy.ndarray, shape=(D, D)"""
    for n in range(len(s) - 1):
        z[n] += np.dot(s[n], Wz)
        z[n] = sig(z[n])

        r[n] += np.dot(s[n], Wr)
        r[n] = sig(r[n])

        h[n] += np.dot(r[n] * s[n], Wh)
        h[n] = np.tanh(h[n])

        s[n + 1] = (1 - z[n]) * h[n] + z[n] * s[n]


@njit
def _gru_dLds(s, z, r, dLds, Wz, Wh, Wr, dz, dh, dr, s_h, one_z):
    """
                        Z_{t} = sigmoid(Uz X_{t} + Wz S_{t-1} + bz)
                        R_{t} = sigmoid(Ur X_{t} + Wr S_{t-1} + br)
                        H_{t} = tanh(Uh X_{t} + Wh (R{t} * S_{t-1}) + bh)
                        S_{t} = (1 - Z{t}) * H{t} + Z{t} * S_{t-1}

    Returns
    --------

        dL / ds(t) =   partial dL / ds(t+1) * ds(t+1) / ds(t)
                     + partial dL / ds(t+1) * ds(t+1) / dz(t) * dz(t) / ds(t)
                     + partial dL / ds(t+1) * ds(t+1) / dh(t) * dh(t) / ds(t)
                     + partial dL / ds(t+1) * ds(t+1) / dh(t) * dh(t) / dr(t) * dr(t) / ds(t)
    """
    dLdh = dot(dLds * one_z * dh, Wh)

    out = z * dLds
    out += dot(dLds * s_h * dz, Wz)
    out += dLdh * r
    out += dot(dLdh * s * dr, Wr)

    return out


@njit
def _gru_bptt(
    X, dLds, s, z, r, Wz, Wh, Wr, dz, dh, dr, s_h, one_z, bp_lim, old_dLds=None
):
    Wz, Wh, Wr = Wz.T, Wh.T, Wr.T
    bptt = bp_lim < len(X) - 1
    if bptt:  # pragma: no cover
        old_dLds = np.zeros_like(dLds)

    for i in range(bp_lim):
        #  dL(t) / ds(t) + dL(t+1) / ds(t)
        if bptt:  # pragma: no cover
            source_index = slice(1, len(dLds) - i)
            target_index = slice(None, len(dLds) - (i + 1))
            dt = dLds[source_index] - old_dLds[source_index]
            old_dLds = np.copy(dLds)
        else:  # no backprop truncation
            source_index = slice(len(dLds) - (i + 1), len(dLds) - i)
            target_index = slice(len(dLds) - (i + 2), len(dLds) - (i + 1))
            dt = dLds[source_index]

        dLds[target_index] += _gru_dLds(
            s[source_index],
            z[source_index],
            r[source_index],
            dt,
            Wz,
            Wh,
            Wr,
            dz[source_index],
            dh[source_index],
            dr[source_index],
            s_h[source_index],
            one_z[source_index],
        )


def _backprop(var, grad):  # pragma: no cover
    if not var.constant:
        if var._grad is None:
            var._grad = np.asarray(grad)
        else:
            var._grad += grad


class GRUnit(Operation):
    def __call__(
        self, X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, s0=None, bp_lim=None, dropout=0.0
    ):
        if bp_lim is not None:
            assert isinstance(bp_lim, Integral) and 0 <= bp_lim < len(X)
        assert 0.0 <= dropout < 1.0
        self._dropout = dropout
        self.bp_lim = bp_lim if bp_lim is not None else len(X) - 1

        self.X = X  # type: Tensor  # shape=(T, N, C)

        self.Uz = Uz  # type: Tensor  # shape=(C, D)
        self.Wz = Wz  # type: Tensor  # shape=(D, D)
        self.bz = bz  # type: Tensor  # shape=(D,)

        self.Ur = Ur  # type: Tensor  # shape=(C, D)
        self.Wr = Wr  # type: Tensor  # shape=(D, D)
        self.br = br  # type: Tensor  # shape=(D,)

        self.Uh = Uh  # type: Tensor  # shape=(C, D)
        self.Wh = Wh  # type: Tensor  # shape=(D, D)
        self.bh = bh  # type: Tensor  # shape=(D,)

        self.variables = (
            self.X,
            self.Uz,
            self.Wz,
            self.bz,
            self.Ur,
            self.Wr,
            self.br,
            self.Uh,
            self.Wh,
            self.bh,
        )

        self.type = max(t.dtype for t in self.variables)

        T, N, C = X.shape
        (D,) = bz.shape

        seq = self.X.data

        # t starts at 0 for S; all other sequences begin at t = 1
        out = np.zeros((T + 1, N, D), dtype=self.type)

        if s0 is not None:
            out[0] = s0.data if isinstance(s0, Tensor) else s0

        # compute all contributions to Z, R, H from the input sequence
        # shape: T, N, D
        z = np.tensordot(seq, self.Uz.data, [[-1], [0]]).astype(self.type, copy=False)
        r = np.tensordot(seq, self.Ur.data, [[-1], [0]]).astype(self.type, copy=False)
        h = np.tensordot(seq, self.Uh.data, [[-1], [0]]).astype(self.type, copy=False)

        if dropout:
            p = 1 - dropout
            # For Uz/Ur/Uh: a dropout mask is generated for each datum and is applied uniformly across T
            self._dropUz, self._dropUr, self._dropUh = (
                np.random.binomial(1, p, size=(3, 1, N, D)) / p
            )
            self._dropWz, self._dropWr, self._dropWh = (
                np.random.binomial(1, p, size=(3, D, D)) / p
            )

            z *= self._dropUz
            r *= self._dropUr
            h *= self._dropUh

            Wz = (self._dropWz * self.Wz.data).astype(self.type, copy=False)
            Wr = (self._dropWr * self.Wr.data).astype(self.type, copy=False)
            Wh = (self._dropWh * self.Wh.data).astype(self.type, copy=False)

        else:
            self._dropUz, self._dropUr, self._dropUh = None, None, None
            self._dropWz, self._dropWr, self._dropWh = None, None, None
            Wz = self.Wz.data.astype(self.type, copy=False)
            Wr = self.Wr.data.astype(self.type, copy=False)
            Wh = self.Wh.data.astype(self.type, copy=False)

        z += bz.data.astype(self.type, copy=False)  # X Uz + bz
        r += br.data.astype(self.type, copy=False)  # X Ur + br
        h += bh.data.astype(self.type, copy=False)  # X Uh + bh

        _gru_layer(out, z, r, h, Wz, Wr, Wh)

        self._z = z
        self._r = r
        self._h = h

        return out

    def backward_var(self, grad, index, **kwargs):
        raise SkipGradient("Gradient computed in GRU.backward()")

    def backward(self, grad, **kwargs):
        hidden_seq = self._hidden_seq()
        if hidden_seq is None:  # pragma: no cover
            assert False, "should be unreachable"

        s = hidden_seq.data[:-1]
        z = self._z
        r = self._r
        h = self._h

        dLds = grad[1:].astype(self.type, copy=False)

        const = {"1 - h**2": d_tanh(h), "z*(1 - z)": d_sig(z), "r*(1 - r)": d_sig(r)}

        if self._dropout:
            Wz = (self._dropWz * self.Wz.data).astype(self.type, copy=False)
            Wr = (self._dropWr * self.Wr.data).astype(self.type, copy=False)
            Wh = (self._dropWh * self.Wh.data).astype(self.type, copy=False)
        else:
            Wz = self.Wz.data.astype(self.type, copy=False)
            Wr = self.Wr.data.astype(self.type, copy=False)
            Wh = self.Wh.data.astype(self.type, copy=False)

        const["s - h"] = s - h
        const["1 - z"] = 1 - z

        _gru_bptt(
            self.X.data,
            dLds,
            s,
            z,
            r,
            Wz,
            Wh,
            Wr,
            const["z*(1 - z)"],
            const["1 - h**2"],
            const["r*(1 - r)"],
            const["s - h"],
            const["1 - z"],
            self.bp_lim,
        )

        zgrad = dLds * const["s - h"]  # dL / dz
        hgrad = dLds * const["1 - z"]  # dL / dh
        rgrad = dot(const["1 - h**2"] * hgrad, Wh.T) * s  # dL / dr

        hidden_seq._grad = dLds

        if not (self.Uz.constant and self.Wz.constant and self.bz.constant):
            dz = zgrad * const["z*(1 - z)"]
        # backprop through Wz
        if not self.Wz.constant:
            dWz = np.tensordot(s, dz, ([0, 1], [0, 1]))
            if self._dropout:
                dWz *= self._dropWz
            _backprop(
                self.Wz, dWz.astype(self.Wz.dtype, copy=False)
            )  # self.Wz.backward(dWz, **kwargs)
        # backprop through bz
        if not self.bz.constant:
            _backprop(self.bz, dz.sum(axis=(0, 1), dtype=self.bz.dtype))
        # backprop through bz
        if not self.Uz.constant:
            if self._dropout:
                dz *= (
                    self._dropUz
                )  # IMPORTANT augmented update: this must come after Wz and bz backprop
            _backprop(
                self.Uz,
                np.tensordot(self.X.data, dz, ([0, 1], [0, 1])).astype(
                    self.Uz.dtype, copy=False
                ),
            )

        if not (self.Ur.constant and self.Wr.constant and self.br.constant):
            dr = rgrad * const["r*(1 - r)"]
        # backprop through Wr
        if not self.Wr.constant:
            dWr = np.tensordot(s, dr, ([0, 1], [0, 1]))
            if self._dropout:
                dWr *= self._dropWr
            _backprop(self.Wr, dWr.astype(self.Wr.dtype, copy=False))
        # backprop through br
        if not self.br.constant:
            _backprop(
                self.br, dr.sum(axis=(0, 1), dtype=self.br.dtype)
            )  # self.br.backward(dr.sum(axis=(0, 1)), **kwargs)
        # backprop through Ur
        if not self.Ur.constant:
            if self._dropout:
                dr *= (  # pragma: no cover
                    self._dropUr
                )  # IMPORTANT augmented update: this must come after Wr and br backprop
            _backprop(
                self.Ur,
                np.tensordot(self.X.data, dr, ([0, 1], [0, 1])).astype(
                    self.Ur.dtype, copy=False
                ),
            )

        if not (self.Uh.constant and self.Wh.constant and self.bh.constant):
            dh = hgrad * const["1 - h**2"]
        # backprop through Wh
        if not self.Wh.constant:
            dWh = np.tensordot((s * r), dh, ([0, 1], [0, 1]))
            if self._dropout:
                dWh *= self._dropWh
            _backprop(
                self.Wh, dWh.astype(self.Wh.dtype, copy=False)
            )  # self.Wh.backward(dWh, **kwargs)
        # backprop through bh
        if not self.bh.constant:
            _backprop(
                self.bh, dh.sum(axis=(0, 1), dtype=self.bh.dtype)
            )  # self.bh.backward(dh.sum(axis=(0, 1)), **kwargs)
        # backprop through Uh
        if not self.Uh.constant:
            if self._dropout:
                dh *= (
                    self._dropUh
                )  # IMPORTANT augmented update: this must come after Wh and bh backprop
            _backprop(
                self.Uh,
                np.tensordot(self.X.data, dh, ([0, 1], [0, 1])).astype(
                    self.Uh.dtype, copy=False
                ),
            )

        # backprop through X
        if not self.X.constant:
            tmp = dLds * const["1 - z"] * const["1 - h**2"]
            if not self._dropout:
                dLdX = np.dot(
                    (dLds * const["s - h"]) * const["z*(1 - z)"], self.Uz.data.T
                )
                dLdX += np.dot(tmp, self.Uh.data.T)
                dLdX += np.dot(
                    np.dot(tmp, Wh.T) * s * const["r*(1 - r)"], self.Ur.data.T
                )
            else:
                dLdX = np.dot(
                    (self._dropUz * (dLds * const["s - h"]) * const["z*(1 - z)"]),
                    self.Uz.data.T,
                )
                dLdX += np.dot(self._dropUh * tmp, self.Uh.data.T)
                dLdX += np.dot(
                    self._dropUr * (dot(tmp, Wh.T) * s * const["r*(1 - r)"]),
                    self.Ur.data.T,
                )
            _backprop(
                self.X, dLdX.astype(self.X.dtype, copy=False)
            )  # self.X.backward(dLdX, **kwargs)

        del self._z
        del self._r
        del self._h

        super().backward(grad)


def gru(
    X,
    Uz,
    Wz,
    bz,
    Ur,
    Wr,
    br,
    Uh,
    Wh,
    bh,
    s0=None,
    bp_lim=None,
    dropout=0.0,
    constant=None,
):
    r"""Performs a forward pass of sequential data through a Gated Recurrent Unit layer, returning
    the 'hidden-descriptors' arrived at by utilizing the trainable parameters as follows::

                Z_{t} = sigmoid(X_{t} Uz + S_{t-1} Wz + bz)
                R_{t} = sigmoid(X_{t} Ur + S_{t-1} Wr + br)
                H_{t} =    tanh(X_{t} Uh + (R{t} * S_{t-1}) Wh + bh)
                S_{t} = (1 - Z{t}) * H{t} + Z{t} * S_{t-1}

    Parameters
    ----------
    X : array_like, shape=(T, N, C)
       The sequential data to be passed forward.

    Uz : array_like, shape=(C, D)
       The weights used to map sequential data to its hidden-descriptor representation

    Wz : array_like, shape=(D, D)
        The weights used to map a hidden-descriptor to a hidden-descriptor.

    bz : array_like, shape=(D,)
       The biases used to scale a hidden-descriptor.

    Ur : array_like, shape=(C, D)
       The weights used to map sequential data to its hidden-descriptor representation

    Wr : array_like, shape=(D, D)
        The weights used to map a hidden-descriptor to a hidden-descriptor.

    br : array_like, shape=(D,)
       The biases used to scale a hidden-descriptor.

    Uh : array_like, shape=(C, D)
       The weights used to map sequential data to its hidden-descriptor representation

    Wh : array_like, shape=(D, D)
        The weights used to map a hidden-descriptor to a hidden-descriptor.

    bh : array_like, shape=(D,)
       The biases used to scale a hidden-descriptor.

    s0 : Optional[array_like], shape=(N, D)
        The 'seed' hidden descriptors to feed into the RNN. If None, a Tensor
        of zeros of shape (N, D) is created.

    bp_lim : Optional[int]
        *This feature is experimental and is currently untested*.
        The (non-zero) limit of the depth of back propagation through time to be
        performed. If `None` back propagation is passed back through the entire sequence.

        E.g. `bp_lim=3` will propagate gradients only up to 3 steps backward through the
        recursive sequence.

    dropout : float (default=0.), 0 <= dropout < 1
        If non-zero, the dropout scheme described in [1]_ is applied. See Notes
        for more details.

    constant : bool, optional (default=False)
        If True, the resulting Tensor is a constant.

    Returns
    -------
    mygrad.Tensor, shape=(T+1, N, D)
        The sequence of 'hidden-descriptors' produced by the forward pass of the RNN.

    Notes
    -----
    - :math:`T` : Sequence length
    - :math:`N` : Batch size
    - :math:`C` : Length of single datum
    - :math:`D` : Length of 'hidden' descriptor

    The GRU system of equations is given by:

    .. math::

                Z_{t} = \sigma (X_{t} U_z + S_{t-1} Wz + bz)

                R_{t} = \sigma (X_{t} U_r + S_{t-1} Wr + br)

                H_{t} =    tanh(X_{t} U_h + (R_{t} * S_{t-1}) W_h + b_h)

                S_{t} = (1 - Z_{t}) * H_{t} + Z_{t} * S_{t-1}

    Following the dropout scheme specified in [1]_, the hidden-hidden weights (Wz/Wr/Wh)
    randomly have their weights dropped prior to forward/back-prop. The input connections
    (via Uz/Ur/Uh) have variational dropout ([2]_) applied to them with a common dropout
    mask across all t. That is three static dropout masks, each with shape-(N,D), are
    applied to

    .. math::
                                          X_{t} U_z

                                          X_{t} U_r

                                          X_{t} U_h
    respectively, for all :math:`t`.

    References
    ----------
    .. [1] S. Merity, et. al. "Regularizing and Optimizing LSTM Language Models",
           arXiv:1708.02182v1, 2017.

    .. [2] Y. Gal, Z. Ghahramani "A Theoretically Grounded Application of Dropout
           in Recurrent Neural Networks" arXiv:1512.05287v5, 2016."""
    if s0 is not None:
        if not isinstance(s0, np.ndarray) and not (
            isinstance(s0, Tensor) and (constant or s0.constant)
        ):
            raise ValueError(
                "GRU does not support non-constant tensors for the initial hidden"
                "state value, `s0`"
            )
    s = Tensor._op(
        GRUnit,
        X,
        Uz,
        Wz,
        bz,
        Ur,
        Wr,
        br,
        Uh,
        Wh,
        bh,
        op_kwargs=dict(s0=s0, bp_lim=bp_lim, dropout=dropout),
        constant=constant,
    )
    try:
        s.creator._hidden_seq = weakref.ref(s)
    except AttributeError:  # pragma: no cover
        # `no-autodiff` mode does not record creator
        pass
    return s
