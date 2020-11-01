import paddle


def normalize_kernel2d(input: paddle.Tensor) -> paddle.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.shape) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.shape))
    norm: paddle.Tensor = input.abs().sum(-1).sum(-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))