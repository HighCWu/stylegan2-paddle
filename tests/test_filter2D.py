import os
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from stylegan2_paddle.kornia.filters import filter2D


class Blur(nn.Layer):
    def __init__(self):
        super().__init__()
        f = paddle.to_tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f.astype(x.dtype)
        f = f.unsqueeze(0).unsqueeze(0) * f.unsqueeze(0).unsqueeze(2)
        return filter2D(x, f, normalized=True)


result_dir = 'test_filter2D_results'
os.makedirs(result_dir, exist_ok=True)

model = Blur()

x = paddle.rand([4,3,128,128])
y = model(x)

for i, t in enumerate(zip(x, y)):
    img = paddle.concat(t, -1).numpy().transpose([1,2,0])*255
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(result_dir, str(i).zfill(3)+'.png'))
