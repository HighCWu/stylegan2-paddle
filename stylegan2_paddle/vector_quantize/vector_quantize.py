import paddle
from paddle import nn
import paddle.nn.functional as F
import paddle.distributed as dist_fn


def ema_inplace(moving_avg, new, decay):
    moving_avg[:] = moving_avg * decay
    moving_avg[:] = decay * moving_avg + (1 - decay) * new

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

class VectorQuantize(nn.Layer):
    def __init__(self, dim, n_embed, decay=0.8, commitment=1., eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        embed = paddle.randn([dim, n_embed])
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', paddle.zeros([n_embed]))
        self.register_buffer('embed_avg', embed.astype(embed.dtype))

    def forward(self, input):
        dtype = input.dtype
        flatten = input.reshape([-1, self.dim])
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten.transpose([0, 1]).matmul(self.embed)
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = (-dist).argmax(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).astype(dtype)
        embed_ind = embed_ind.reshape(input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose([1, 0]), padding_idx=-1)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose([1, 0]).matmul(embed_onehot)

            if dist_fn.get_world_size() > 1:
                dist_fn.all_reduce(embed_onehot_sum)
                dist_fn.all_reduce(embed_sum)

            ema_inplace(self.cluster_size, embed_onehot_sum, self.decay)
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed[:] = embed_normalized

        loss = F.mse_loss(quantize.detach(), input) * self.commitment
        quantize = input + (quantize - input).detach()
        return quantize, embed_ind, loss