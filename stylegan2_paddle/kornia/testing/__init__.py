import paddle


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor.
    """
    if not isinstance(obj, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}".format(type(obj)))