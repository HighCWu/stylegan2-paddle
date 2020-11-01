import os
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from stylegan2_paddle.linear_attention_transformer import ImageLinearAttention

model = ImageLinearAttention(64, norm_queries = True)

x = paddle.randn([4,64,64,64])
y = model(x)
print(y.shape)