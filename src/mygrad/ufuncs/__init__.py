"""
Exposes MyGrad's ufunc types (which are distinct from the Operation subclasses that are provided to
implement the machinery for backpropagation through a particular operation). Also provides a wrapper
for creating a ufunc with a specified name and signature from a given function stub."""

from ._ufunc_creators import MyGradBinaryUfunc, MyGradUnaryUfunc, ufunc_creator

__all__ = ["ufunc_creator", "MyGradUnaryUfunc", "MyGradBinaryUfunc"]
