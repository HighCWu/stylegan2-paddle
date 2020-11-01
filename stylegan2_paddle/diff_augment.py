import paddle
import paddle.nn.functional as F

def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x

def rand_brightness(x):
    x = x + (paddle.rand([x.shape[0], 1, 1, 1], dtype=x.dtype) - 0.5)
    return x

def rand_saturation(x):
    x_mean = x.mean(1, keepdim=True)
    x = (x - x_mean) * (paddle.rand([x.shape[0], 1, 1, 1], dtype=x.dtype) * 2) + x_mean
    return x

def rand_contrast(x):
    x_mean = x.mean([1, 2, 3], keepdim=True)
    x = (x - x_mean) * (paddle.rand([x.shape[0], 1, 1, 1], dtype=x.dtype) + 0.5) + x_mean
    return x

def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.shape[2] * ratio + 0.5), int(x.shape[3] * ratio + 0.5)
    translation_x = paddle.randint(-shift_x, shift_x + 1, shape=[x.shape[0], 1, 1])
    translation_y = paddle.randint(-shift_y, shift_y + 1, shape=[x.shape[0], 1, 1])
    grid_batch, grid_x, grid_y = paddle.meshgrid(
        paddle.arange(x.shape[0], dtype='int64'),
        paddle.arange(x.shape[2], dtype='int64'),
        paddle.arange(x.shape[3], dtype='int64'),
    )
    grid_x = paddle.clip((grid_x + translation_x + 1).astype(x.dtype), 0, x.shape[2] + 1).astype('int64')
    grid_y = paddle.clip((grid_y + translation_y + 1).astype(x.dtype), 0, x.shape[3] + 1).astype('int64')
    x_pad = F.pad(x, [1, 1, 1, 1])
    
    # TODO: Current version paddle doesn't support int64 Tensors indices
    # x = x_pad.transpose([0, 2, 3, 1])[grid_batch, grid_x, grid_y].transpose([0, 3, 1, 2])
    indices = paddle.stack([grid_batch, grid_x, grid_y], -1)
    x = x_pad.transpose([0, 2, 3, 1]).gather_nd(indices).transpose([0, 3, 1, 2])

    return x

def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.shape[2] * ratio + 0.5), int(x.shape[3] * ratio + 0.5)
    offset_x = paddle.randint(0, x.shape[2] + (1 - cutout_size[0] % 2), shape=[x.shape[0], 1, 1])
    offset_y = paddle.randint(0, x.shape[3] + (1 - cutout_size[1] % 2), shape=[x.shape[0], 1, 1])
    
    # TODO: Current version paddle doesn't support int64 Tensors indices
    # grid_batch, grid_x, grid_y = paddle.meshgrid(
    #     paddle.arange(x.shape[0], dtype='int64'),
    #     paddle.arange(cutout_size[0], dtype='int64'),
    #     paddle.arange(cutout_size[1], dtype='int64'),
    # )
    # grid_x = paddle.clip((grid_x + offset_x - cutout_size[0] // 2).astype(x.dtype), min=0, max=x.shape[2] - 1).astype('int64')
    # grid_y = paddle.clip((grid_y + offset_y - cutout_size[1] // 2).astype(x.dtype), min=0, max=x.shape[3] - 1).astype('int64')
    # mask = paddle.ones([x.shape[0], x.shape[2], x.shape[3]], dtype=x.dtype)
    # mask[grid_batch, grid_x, grid_y] = 0
    grid_batch, grid_x, grid_y = paddle.meshgrid(
        paddle.arange(x.shape[0], dtype='int64'),
        paddle.arange(x.shape[2], dtype='int64'),
        paddle.arange(x.shape[3], dtype='int64'),
    )
    grid_x = grid_x + offset_x - cutout_size[0] // 2
    grid_y = grid_y + offset_y - cutout_size[1] // 2
    mask = 1 - ((grid_x >= 0).astype(x.dtype) * 
                (grid_x < cutout_size[0]).astype(x.dtype) * 
                (grid_y >= 0).astype(x.dtype) *
                (grid_y < cutout_size[1]).astype(x.dtype)).astype(x.dtype)

    x = x * mask.unsqueeze(1).detach()
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
