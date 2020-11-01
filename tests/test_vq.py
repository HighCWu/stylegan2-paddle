import os
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from stylegan2_paddle.vector_quantize import VectorQuantize

model = VectorQuantize(64, 512)

cluster_size1 = model.cluster_size.astype(model.cluster_size.dtype)
embed_avg1 = model.embed_avg.astype(model.embed_avg.dtype)
embed1 = model.embed.astype(model.embed.dtype)

x = paddle.randn([4,64,64,64])
model(x)

cluster_size2 = model.cluster_size.astype(model.cluster_size.dtype)
embed_avg2 = model.embed_avg.astype(model.embed_avg.dtype)
embed2 = model.embed.astype(model.embed.dtype)

print('Expect not zero:', (cluster_size2 - cluster_size1).mean().numpy())
print('Expect not zero:', (embed_avg2 - embed_avg1).mean().numpy())
print('Expect not zero:', (embed2 - embed1).mean().numpy())