def reduce_broadcast(grad, outshape):
    if grad.shape == outshape:
        return grad

    if grad.ndim != len(outshape):
        assert grad.ndim > len(outshape)
        grad = grad.sum(axis=tuple(range(grad.ndim - len(outshape))))

    keepdims = tuple(n for n, i in enumerate(grad.shape) if i != outshape[n])
    if keepdims:
        grad = grad.sum(axis=keepdims, keepdims=True)

    return grad