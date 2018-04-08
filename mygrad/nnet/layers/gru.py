from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor
from numbers import Integral
import numpy as np

from numba import njit, vectorize


@vectorize(['int32(int32)',
            'int64(int64)',
            'float32(float32)',
            'float64(float64)'], nopython=True)
def sig(f):
    """
    Calculates a sigmoid function
    """
    return 1 / (1 + np.exp(-f))


@vectorize(['int32(int32)',
            'int64(int64)',
            'float32(float32)',
            'float64(float64)'], nopython=True)
def d_sig(f):
    """
    Calculates the derivative of a sigmoid function
    """
    return f * (1 - f)


@vectorize(['int32(int32)',
            'int64(int64)',
            'float32(float32)',
            'float64(float64)'], nopython=True)
def d_tanh(f):
    """
    Calculates the derivative of a tanh function
    """
    return 1 - f ** 2


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
def _gru_layer(s, z, r, h, Wz, Wr, Wh, bz, br, bh):
    for n in range(len(s) - 1):
        z[n] += np.dot(s[n], Wz) + bz
        z[n] = sig(z[n])

        r[n] += np.dot(s[n], Wr) + br
        r[n] = sig(r[n])

        h[n] += np.dot(r[n] * s[n], Wh) + bh
        h[n] = np.tanh(h[n])

        s[n + 1] = (1 - z[n]) * h[n] + z[n] * s[n]


@njit
def _gru_layer_dropout(s, z, r, h, Wz, Wr, Wh, bz, br, bh, dropz, dropr, droph):
    for n in range(len(s) - 1):
        z[n] += np.dot(s[n], Wz) + bz
        z[n] = (1 / (1 + np.exp(-z[n])))

        r[n] += np.dot(s[n], Wr) + br
        r[n] = (1 / (1 + np.exp(-r[n])))

        h[n] += np.dot(dropr[n] * r[n] * s[n], Wh) + bh
        h[n] = np.tanh(h[n])

        zd = dropz[n] * z[n]
        s[n + 1] = (1 - zd) * droph[n] * h[n] + zd * s[n]


@njit
def _gru_dLds(s, z, r, dLds, Wz, Wh, Wr, dz, dh, dr, s_h, one_z):
    """
                        Z_{t} = sigmoid(Uz X_{t} + Wz S_{t-1} + bz)
                        R_{t} = sigmoid(Ur X_{t} + Wr S_{t-1} + br)
                        H_{t} = tanh(Uh X_{t} + Wh (R{t} * S_{t-1}) + bh)
                        S_{t} = (1 - Z{t}) * H{t} + Z{t} * S_{t-1}

    Returns
    --------
        partial dL / ds(t+1) * ds(t+1) / ds(t) +
        partial dL / ds(t+1) * ds(t+1) / dz(t) * dz(t) / ds(t) +
        partial dL / ds(t+1) * ds(t+1) / dh(t) * dh(t) / ds(t) +
        partial dL / ds(t+1) * ds(t+1) / dh(t) * dh(t) / dr(t) * dr(t) / ds(t)
    """
    dLdh = dot(dLds * one_z * dh, Wh)

    out = z * dLds
    out += dot(dLds * s_h * dz, Wz)
    out += dLdh * r
    out += dot(dLdh * s * dr, Wr)

    return out


@njit
def _gru_bptt(X, dLds, s, z, r, Wz, Wh, Wr, dz, dh, dr, s_h, one_z, bp_lim, old_dLds=None):
    Wz, Wh, Wr = Wz.T, Wh.T, Wr.T
    bptt = bp_lim < len(X) - 1
    if bptt:
        old_dLds = np.zeros_like(dLds)

    for i in range(bp_lim):
        #  dL(t) / ds(t) + dL(t+1) / ds(t)
        if bptt:
            source_index = slice(1, len(dLds) - i)
            target_index = slice(None, len(dLds) - (i + 1))
            dt = dLds[source_index] - old_dLds[source_index]
            old_dLds = np.copy(dLds)
        else:  # no backprop truncation
            source_index = slice(len(dLds) - (i + 1), len(dLds) - i)
            target_index = slice(len(dLds) - (i + 2), len(dLds) - (i + 1))
            dt = dLds[source_index]

        dLds[target_index] += _gru_dLds(s[source_index],
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
                                        one_z[source_index])


import mygrad
class GRUnit(Operation):
    def __call__(self, X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, s0=None, bp_lim=None, dropout=0.):
        if bp_lim is not None:
            assert isinstance(bp_lim, Integral) and 0 <= bp_lim < len(X)
        assert 0. <= dropout < 1.
        self._dropout = dropout
        self.bp_lim = bp_lim if bp_lim is not None else len(X) - 1

        self.X = X    # type: Tensor  # shape=(T, N, C)

        self.Uz = Uz  # type: Tensor  # shape=(C, D)
        self.Wz = Wz  # type: Tensor  # shape=(D, D)
        self.bz = bz  # type: Tensor  # shape=(D,)

        self.Ur = Ur  # type: Tensor  # shape=(C, D)
        self.Wr = Wr  # type: Tensor  # shape=(D, D)
        self.br = br  # type: Tensor  # shape=(D,)

        self.Uh = Uh  # type: Tensor  # shape=(C, D)
        self.Wh = Wh  # type: Tensor  # shape=(D, D)
        self.bh = bh  # type: Tensor  # shape=(D,)

        T, N, C = X.shape
        D, = bz.shape

        seq = self.X.data

        # t starts at 0 for S; all other sequences begin at t = 1
        out = np.zeros((T + 1, N, D))

        if s0 is not None:
            out[0] = s0.data if isinstance(s0, Tensor) else s0

        z = dot(seq, self.Uz.data)
        r = dot(seq, self.Ur.data)
        h = dot(seq, self.Uh.data)

        if not dropout:
            self._dropz = None
            self._dropr = None
            self._droph = None
            _gru_layer(out, z, r, h,
                       self.Wz.data, self.Wr.data, self.Wh.data,
                       self.bz.data, self.br.data, self.bh.data)
        else:
            p = 1 - dropout
            self._dropz = np.random.binomial(1, p, size=z.shape) / p
            self._dropr = np.random.binomial(1, p, size=r.shape) / p
            self._droph = np.random.binomial(1, p, size=h.shape) / p

            _gru_layer_dropout(out, z, r, h,
                               self.Wz.data, self.Wr.data, self.Wh.data,
                               self.bz.data, self.br.data, self.bh.data,
                               self._dropz, self._dropr, self._droph)

        self._hidden_seq = Tensor(out, _creator=self)
        self._z = Tensor(z, _creator=self)
        self._r = Tensor(r, _creator=self)
        self._h = Tensor(h, _creator=self)

        return self._hidden_seq

    def backward(self, grad, **kwargs):
        if all(i.constant for i in [self.X,
                                    self.Uz, self.Wz, self.bz,
                                    self.Ur, self.Wr, self.br,
                                    self.Uh, self.Wh, self.bh]):
            return None

        s = self._hidden_seq.data[:-1]
        z = self._z.data
        r = self._r.data
        h = self._h.data

        dLds = grad[1:]

        const = {"1 - h**2": d_tanh(h),
                 "z*(1 - z)": d_sig(z),
                 "r*(1 - r)": d_sig(r)}

        if self._dropout:
            const["1 - h**2"] *= self._droph
            const["r*(1 - r)"] *= self._dropr
            const["z*(1 - z)"] *= self._dropz
            h *= self._droph
            r *= self._dropr
            z *= self._dropz

        const["s - h"] = s - h
        const["1 - z"] = 1 - z

        _gru_bptt(self.X.data, dLds, s, z, r,
                   self.Wz.data,
                   self.Wh.data,
                   self.Wr.data,
                   const["z*(1 - z)"],
                   const["1 - h**2"],
                   const["r*(1 - r)"],
                   const["s - h"],
                   const["1 - z"],
                   self.bp_lim)

        zgrad = dLds * const["s - h"]   # dL / dz
        hgrad = dLds * const["1 - z"]   # dL / dh
        rgrad = dot(const["1 - h**2"] * hgrad, self.Wh.data.T) * s  # dL / dr

        self._hidden_seq.grad = dLds
        self._z.grad = zgrad
        self._r.grad = rgrad
        self._h.grad = hgrad

        if any(not const for const in (self.Uz.constant, self.Wz.constant, self.bz.constant)):
            dz = zgrad * const["z*(1 - z)"]

        if not self.Uz.constant:
            self.Uz.backward(np.tensordot(self.X.data, dz, ([0, 1], [0, 1])), **kwargs)
        if not self.Wz.constant:
            self.Wz.backward(np.tensordot(s, dz, ([0, 1], [0, 1])), **kwargs)
        if not self.bz.constant:
            self.bz.backward(dz.sum(axis=(0, 1)), **kwargs)

        if any(not const for const in (self.Ur.constant, self.Wr.constant, self.br.constant)):
            dr = rgrad * const["r*(1 - r)"]

        if not self.Ur.constant:
            self.Ur.backward(np.tensordot(self.X.data, dr, ([0, 1], [0, 1])), **kwargs)
        if not self.Wr.constant:
            self.Wr.backward(np.tensordot(s, dr, ([0, 1], [0, 1])), **kwargs)
        if not self.br.constant:
            self.br.backward(dr.sum(axis=(0, 1)), **kwargs)

        if any(not const for const in (self.Uh.constant, self.Wh.constant, self.bh.constant)):
            dh = hgrad * const["1 - h**2"]

        if not self.Uh.constant:
            self.Uh.backward(np.tensordot(self.X.data, dh, ([0, 1], [0, 1])), **kwargs)
        if not self.Wh.constant:
            self.Wh.backward(np.tensordot((s * r), dh, ([0, 1], [0, 1])), **kwargs)
        if not self.bh.constant:
            self.bh.backward(dh.sum(axis=(0, 1)), **kwargs)

        if not self.X.constant:
            tmp = dLds * const["1 - z"] * const["1 - h**2"]

            dLdX = dot((dLds * const["s - h"]) * const["z*(1 - z)"], self.Uz.data.T)
            dLdX += dot(tmp, self.Uh.data.T)
            dLdX += dot(dot(tmp, self.Wh.data.T) * s * const["r*(1 - r)"], self.Ur.data.T)

            self.X.backward(dLdX, **kwargs)

    def null_gradients(self):
        """ Back-propagates `None` to the gradients of the operation's input Tensors."""
        for x in [self.X,
                  self.Uz, self.Wz, self.bz,
                  self.Ur, self.Wr, self.br,
                  self.Uh, self.Wh, self.bh]:
            x.null_gradients()


def gru(X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, s0=None, bp_lim=None, dropout=0.):
    """ Performs a forward pass of sequential data through a Gated Recurrent Unit layer, returning
        the 'hidden-descriptors' arrived at by utilizing the trainable parameters as follows:

                    Z_{t} = drop_z * sigmoid(Uz X_{t} + Wz S_{t-1} + bz)
                    R_{t} = drop_r * sigmoid(Ur X_{t} + Wr S_{t-1} + br)
                    H_{t} = drop_h * tanh(Uh X_{t} + Wh (R{t} * S_{t-1}) + bh)
                    S_{t} = (1 - Z{t}) * H{t} + Z{t} * S_{t-1}

        Parameters
        ----------
        X : mygrad.Tensor, shape=(T, N, C)
           The sequential data to be passed forward.

        Uz/Ur/Uh : mygrad.Tensor, shape=(C, D)
           The weights used to map sequential data to its hidden-descriptor representation

        Wz/Wr/Wh : mygrad.Tensor, shape=(D, D)
            The weights used to map a hidden-descriptor to a hidden-descriptor.

        bz/br/bh : mygrad.Tensor, shape=(D,)
           The biases used to scale a hidden-descriptor.

        s0 : Optional[mygrad.Tensor, numpy.ndarray], shape=(N, D)
            The 'seed' hidden descriptors to feed into the RNN. If None, a Tensor
            of zeros of shape (N, D) is created.

        bp_lim : Optional[int]
            The (non-zero) limit of the depth of back propagation through time to be
            performed. If `None` back propagation is passed back through the entire sequence.

            E.g. `bp_lim=3` will propagate gradients only up to 3 steps backward through the
            recursive sequence.

        dropout : float (default=0.), 0 <= dropout < 1
            If non-zero, the expected proportion of a layer's outputs to be set to zero. Applied
            to the layers Z, R, and H, - *not* S.

            These layers are also scaled by 1 / dropout, such that the test-time forward pass
            of the gru-layer can be executed without dropout, and with no additional scaling.
            (only if `dropout` is non-zero)

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
    s = Tensor._op(GRUnit, X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, op_kwargs=dict(s0=s0, bp_lim=bp_lim, dropout=dropout))
    s.creator._hidden_seq = s
    return s
