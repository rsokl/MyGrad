import mygrad as mg
from mygrad.typing import ArrayLike


def check_consistent_grad_dtype(*args: ArrayLike):
    """Raises assertion error if `t.grad.dtype` does not match `t.dtype`
    for variable tensors."""
    for item in args:
        if not isinstance(item, mg.Tensor):
            continue
        elif item.constant:
            assert item.grad is None
        else:
            assert item.grad.dtype == item.dtype
