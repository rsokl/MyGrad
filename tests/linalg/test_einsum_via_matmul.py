"""Tests for computing two-operand einsums via matmul (see `mygrad.linalg.ops`)"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mygrad as mg
import mygrad.linalg.ops as linalg_ops
from mygrad.linalg.ops import _einsum_via_matmul

# expression, shape-a, shape-b
SUPPORTED = [
    ("ij,jk->ik", (3, 4), (4, 5)),  # matmul
    ("ij,jk->ki", (3, 4), (4, 5)),  # transposed output
    ("ik,jk->ij", (3, 4), (5, 4)),  # a @ b.T
    ("bij,bjk->bik", (2, 3, 4), (2, 4, 5)),  # batched
    ("bqd,bkd->bqk", (2, 3, 4), (2, 5, 4)),  # attention scores
    ("ijk,jkl->il", (2, 3, 4), (3, 4, 5)),  # multiple contracted labels
    ("ijk,k->ij", (2, 3, 4), (4,)),  # matrix-vector
    ("k,kj->j", (4,), (4, 5)),  # vector-matrix
    ("ijk,kl->li", (2, 3, 4), (4, 5)),  # `j` is summed out of `a` alone
    ("ij,jk->i", (3, 4), (4, 5)),  # `k` is summed out of `b` alone
    ("bij,jk->bik", (2, 3, 4), (4, 5)),  # `b` broadcast over a batch
    ("ij,jkl->ikl", (3, 4), (4, 2, 5)),
]


@pytest.fixture
def always_use_matmul(monkeypatch):
    monkeypatch.setattr(linalg_ops, "_MIN_MATMUL_WORK", 0)


def _forward_and_grads(expr, x, y):
    a, b = mg.tensor(x), mg.tensor(y)
    out = mg.einsum(expr, a, b)
    out.backward(np.arange(out.size, dtype=float).reshape(out.shape))
    return out.data, a.grad, b.grad


@pytest.mark.parametrize("expr, shape_a, shape_b", SUPPORTED)
def test_supported_contractions_use_matmul(expr, shape_a, shape_b, always_use_matmul):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=shape_a), rng.normal(size=shape_b)
    ins, out = expr.split("->")
    lhs, rhs = ins.split(",")
    result = _einsum_via_matmul(lhs, rhs, out, x, y)
    assert result is not None
    assert_allclose(result, np.einsum(expr, x, y))


@pytest.mark.parametrize("expr, shape_a, shape_b", SUPPORTED)
def test_matmul_path_matches_einsum_path(
    expr, shape_a, shape_b, monkeypatch: pytest.MonkeyPatch
):
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=shape_a), rng.normal(size=shape_b)

    with monkeypatch.context() as m:
        m.setattr(linalg_ops, "_einsum_via_matmul", lambda *args: None)
        expected = _forward_and_grads(expr, x, y)

    with monkeypatch.context() as m:
        m.setattr(linalg_ops, "_MIN_MATMUL_WORK", 0)
        actual = _forward_and_grads(expr, x, y)

    for exp, act in zip(expected, actual):
        assert_allclose(act, exp, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "expr, a, b",
    [
        ("ij,jk->ik", np.ones((3, 4), dtype=int), np.ones((4, 5), dtype=int)),  # ints
        ("ii,ij->ij", np.ones((4, 4)), np.ones((4, 5))),  # repeated label
        ("ij,jk->ik", np.ones((3, 1)), np.ones((4, 5))),  # broadcast between operands
        ("i,i->", np.ones(4), np.ones(4)),  # dot product
        ("ij,ij->i", np.ones((3, 4)), np.ones((3, 4))),  # batch of dot products
        ("ij,kl->ijkl", np.ones((3, 4)), np.ones((2, 5))),  # outer product
    ],
)
def test_unsupported_contractions_fall_back(expr, a, b, always_use_matmul):
    ins, out = expr.split("->")
    lhs, rhs = ins.split(",")
    assert _einsum_via_matmul(lhs, rhs, out, a, b) is None


def test_small_contractions_fall_back():
    a, b = np.ones((4, 4)), np.ones((4, 4))
    assert _einsum_via_matmul("ij", "jk", "ik", a, b) is None


def test_large_contractions_use_matmul():
    a, b = np.ones((64, 64)), np.ones((64, 64))
    assert _einsum_via_matmul("ij", "jk", "ik", a, b) is not None


def test_explicit_optimize_uses_numpy(monkeypatch: pytest.MonkeyPatch):
    def fail(*args):  # pragma: no cover
        raise AssertionError("matmul path should not be used")

    monkeypatch.setattr(linalg_ops, "_einsum_via_matmul", fail)
    a, b = mg.tensor(np.ones((64, 64))), mg.tensor(np.ones((64, 64)))
    out = mg.einsum("ij,jk->ik", a, b, optimize=True)
    out.backward()
    assert_allclose(out, np.full((64, 64), 64.0))


def test_einsum_writes_to_provided_out_array():
    a, b = mg.tensor(np.ones((64, 64))), mg.tensor(np.ones((64, 64)))
    out = np.zeros((64, 64))
    result = mg.einsum("ij,jk->ik", a, b, out=out)
    assert np.shares_memory(result, out)
    assert_allclose(out, np.full((64, 64), 64.0))
