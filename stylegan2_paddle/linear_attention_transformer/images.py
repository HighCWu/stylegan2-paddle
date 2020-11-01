import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class ImageLinearAttention(nn.Layer):
    def __init__(self, chan, chan_out = None, kernel_size = 1, padding = 0, stride = 1, key_dim = 64, value_dim = 64, heads = 8, norm_queries = True):
        super().__init__()
        self.chan = chan
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.norm_queries = norm_queries

        conv_kwargs = {'padding': padding, 'stride': stride}
        self.to_q = nn.Conv2D(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_k = nn.Conv2D(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_v = nn.Conv2D(chan, value_dim * heads, kernel_size, **conv_kwargs)

        out_conv_kwargs = {'padding': padding}
        self.to_out = nn.Conv2D(value_dim * heads, chan_out, kernel_size, **out_conv_kwargs)

    def forward(self, x, context = None):
        b, c, h, w, k_dim, heads = *x.shape, self.key_dim, self.heads

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        q, k, v = map(lambda t: t.reshape([b, heads, -1, h * w]), (q, k, v))

        q, k = map(lambda x: x * (self.key_dim ** -0.25), (q, k))

        if context is not None:
            context = context.reshape([b, c, 1, -1])
            ck, cv = self.to_k(context), self.to_v(context)
            ck, cv = map(lambda t: t.reshape([b, heads, k_dim, -1]), (ck, cv))
            k = paddle.concat((k, ck), 3)
            v = paddle.concat((v, cv), 3)

        k = F.softmax(k, -1)

        if self.norm_queries:
            q = F.softmax(q, -2)

        # TODO: Current version paddle doesn't support einsum
        # paddle.einsum('bhdn,bhen->bhde', k, v)
        # paddle.einsum('bhdn,bhde->bhen', q, context)
        context = paddle.matmul(k, q.transpose([0,1,3,2]))
        out = paddle.matmul(context.transpose([0,1,3,2]), q) 

        out = out.reshape([b, -1, h, w])
        out = self.to_out(out)
        return out