from ...operations.operation_base import Operation
from ...tensor_base import Tensor
from numbers import Integral
import numpy as np

from numba import njit


@njit
def _gru_layer(s, z, r, h, Wz, Wr, Wh, bz, br, bh):
    for n in range(len(s) - 1):
        z[n] += np.dot(s[n], Wz) + bz
        z[n] = 1 / (1 + np.exp(-z[n]))

        r[n] += np.dot(s[n], Wr) + br
        r[n] = 1 / (1 + np.exp(-r[n]))

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


class GRUnit(Operation):

    def __call__(self, X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, s0=None, bp_lim=None, dropout=0.):
        if bp_lim is not None:
            assert isinstance(bp_lim, Integral) and 0 <= bp_lim < len(X)
        assert 0. <= dropout < 1.
        self._dropout = dropout
        self.bp_lim = bp_lim if bp_lim is not None else len(X) - 1

        self.X = X

        self.Uz = Uz
        self.Wz = Wz
        self.bz = bz

        self.Ur = Ur
        self.Wr = Wr
        self.br = br

        self.Uh = Uh
        self.Wh = Wh
        self.bh = bh

        seq = self.X.data

        z = np.zeros((seq.shape[0], seq.shape[1], self.Uz.shape[-1]))
        r = np.zeros((seq.shape[0], seq.shape[1], self.Ur.shape[-1]))
        h = np.zeros((seq.shape[0], seq.shape[1], self.Uh.shape[-1]))
        out = np.zeros((seq.shape[0] + 1, seq.shape[1], self.Uz.shape[-1]))

        if s0 is not None:
            out[0] = s0.data if isinstance(s0, Tensor) else s0

        np.dot(seq, self.Uz.data, out=z)
        np.dot(seq, self.Ur.data, out=r)
        np.dot(seq, self.Uh.data, out=h)

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

    def _gru_dLds(self, s, z, r, dLds, dz, dh, dr, s_h, one_z):
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
        Wz, Wr, Wh = self.Wz.data, self.Wr.data, self.Wh.data
        dLdh = np.dot(dLds * one_z * dh, Wh.T)

        out = z * dLds
        out += np.dot(dLds * s_h * dz, Wz.T)
        out += dLdh * r
        out += np.dot(dLdh * s * dr, Wr.T)

        return out

    def backward(self, grad):
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
        if self.bp_lim < len(self.X) - 1:
            old_dLds = np.zeros_like(dLds)

        const = {"1 - h**2": 1 - h**2,
                 "z*(1 - z)": z * (1 - z),
                 "r*(1 - r)": r * (1 - r)}

        if self._dropout:
            const["1 - h**2"] *= self._droph
            const["r*(1 - r)"] *= self._dropr
            const["z*(1 - z)"] *= self._dropz
            h *= self._droph
            r *= self._dropr
            z *= self._dropz

        const["s - h"] = s - h
        const["1 - z"] = 1 - z

        for i in range(self.bp_lim):
            #  dL(t) / ds(t) + dL(t+1) / ds(t)
            if self.bp_lim < len(self.X) - 1:
                source_index = slice(1, len(dLds) - i)
                target_index = slice(None, len(dLds) - (i + 1))
                dt = dLds[source_index] - old_dLds[source_index]
                old_dLds = np.copy(dLds)
            else:  # no backprop truncation
                source_index = len(dLds) - (i + 1)
                target_index = len(dLds) - (i + 2)
                dt = dLds[source_index]
                
            dLds[target_index] += self._gru_dLds(s[source_index],
                                                 z[source_index],
                                                 r[source_index],
                                                 dt,
                                                 const["z*(1 - z)"][source_index],
                                                 const["1 - h**2"][source_index],
                                                 const["r*(1 - r)"][source_index],
                                                 const["s - h"][source_index],
                                                 const["1 - z"][source_index])

        zgrad = dLds * (s - h)   # dL / dz
        hgrad = dLds * const["1 - z"]   # dL / dh
        rgrad = np.dot(const["1 - h**2"] * hgrad, self.Wh.data.T) * s  # dL / dr

        self._hidden_seq.grad = dLds
        self._z.grad = zgrad
        self._r.grad = rgrad
        self._h.grad = hgrad

        if any(not const for const in (self.Uz.constant, self.Wz.constant, self.bz.constant)):
            dz = zgrad * const["z*(1 - z)"]

        if not self.Uz.constant:
            self.Uz.backward(np.einsum("ijk, ijl -> kl", self.X.data, dz))
        if not self.Wz.constant:
            self.Wz.backward(np.einsum("ijk, ijl -> kl", s, dz))
        if not self.bz.constant:
            self.bz.backward(dz.sum(axis=(0, 1)))

        if any(not const for const in (self.Ur.constant, self.Wr.constant, self.br.constant)):
            dr = rgrad * const["r*(1 - r)"]

        if not self.Ur.constant:
            self.Ur.backward(np.einsum("ijk, ijl -> kl", self.X.data, dr))
        if not self.Wr.constant:
            self.Wr.backward(np.einsum("ijk, ijl -> kl", s, dr))
        if not self.br.constant:
            self.br.backward(dr.sum(axis=(0, 1)))

        if any(not const for const in (self.Uh.constant, self.Wh.constant, self.bh.constant)):
            dh = hgrad * const["1 - h**2"]

        if not self.Uh.constant:
            self.Uh.backward(np.einsum("ijk, ijl -> kl", self.X.data, dh))
        if not self.Wh.constant:
            self.Wh.backward(np.einsum("ijk, ijl -> kl", (s * r), dh))
        if not self.bh.constant:
            self.bh.backward(dh.sum(axis=(0, 1)))

        if not self.X.constant:
            tmp = dLds * const["1 - z"] * const["1 - h**2"]

            dLdX = np.dot((dLds * const["s - h"]) * const["z*(1 - z)"], self.Uz.data.T)
            dLdX += np.dot(tmp, self.Uh.data.T)
            dLdX += np.dot(np.dot(tmp, self.Wh.data.T) * s * const["r*(1 - r)"], self.Ur.data.T)

            self.X.backward(dLdX)

    def null_gradients(self):
        """ Back-propagates `None` to the gradients of the operation's input Tensors."""
        for x in [self.X, self.Uz, self.Wz, self.bz, self.Ur, self.Wr, self.br, self.Uh, self.Wh, self.bh]:
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

        Uz/Ur/Uh : mygrad.Tensor, shape=(D, C)
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
