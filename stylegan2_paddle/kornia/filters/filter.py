from typing import List

import paddle
import paddle.nn.functional as F

from stylegan2_paddle.kornia.filters.kernels import normalize_kernel2d
import stylegan2_paddle.kornia.testing as testing


def compute_padding(kernel_size: List[int]) -> List[int]:
    """Computes padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pypaddle.org/docs/stable/nn.html#paddle.nn.functional.pad
    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(

    out_padding = []

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding.append(padding)
        out_padding.append(computed_tmp)
    return out_padding


def filter2D(input: paddle.Tensor, kernel: paddle.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> paddle.Tensor:
    r"""Convolve a tensor with a 2d kernel.
    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.
    Args:
        input (paddle.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (paddle.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.
    Return:
        paddle.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.
    Example:
        >>> input = paddle.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = paddle.ones(1, 3, 3)
        >>> kornia.filter2D(input, kernel)
        paddle.tensor([[[[0., 0., 0., 0., 0.]
                        [0., 5., 5., 5., 0.]
                        [0., 5., 5., 5., 0.]
                        [0., 5., 5., 5., 0.]
                        [0., 0., 0., 0., 0.]]]])
    """
    testing.check_is_tensor(input)
    testing.check_is_tensor(kernel)

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: paddle.Tensor = kernel.unsqueeze(1)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand([-1, c, -1, -1])

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding([height, width])

    # TODO: The Op pad3d_grad doesn't have any grad op for gradient penalty in current paddle version
    # input_pad: paddle.Tensor = F.pad(input, padding_shape, mode=border_type)
    input_pad = input

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape([-1, 1, height, width])
    input_pad = input_pad.reshape([-1, tmp_kernel.shape[0], input_pad.shape[-2], input_pad.shape[-1]])

    # convolve the tensor with the kernel.

    # TODO: The Op depthwise_conv2d_grad and pad3d_grad doesn't have any grad op for gradient penalty in current paddle version
    # output = F.conv2d(input_pad, tmp_kernel, groups=tmp_kernel.shape[0], padding=0, stride=1)
    input_pad = input_pad.reshape([-1, 1, input_pad.shape[-2], input_pad.shape[-1]])
    tmp_kernel = tmp_kernel[:1]
    output = F.conv2d(input_pad, tmp_kernel, padding=padding_shape, stride=1)
    
    return output.reshape([b, c, h, w])
