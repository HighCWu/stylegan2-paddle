import os
import sys
import math
import fire
import json

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import paddle
from paddle import nn
from paddle import io
from paddle.optimizer import Adam
import paddle.nn.functional as F
from paddle import grad as paddle_grad
from paddle import DataParallel as DDP

from stylegan2_paddle.kornia.filters import filter2D

from stylegan2_paddle import utils
from paddle.vision import transforms
from stylegan2_paddle.version import __version__
from stylegan2_paddle.diff_augment import DiffAugment

from stylegan2_paddle.fid import fid_score

from stylegan2_paddle.vector_quantize import VectorQuantize
from stylegan2_paddle.linear_attention_transformer import ImageLinearAttention

from PIL import Image
from pathlib import Path

from paddle import amp
APEX_AVAILABLE = True

assert paddle.fluid.core.get_cuda_device_count() > 0, 'You need to have an Nvidia GPU with CUDA installed.'

num_cores = multiprocessing.cpu_count()

# constants

EXTS = ['jpg', 'jpeg', 'png']
EPS = 1e-8
EVALUATE_EVERY = 1000
CALC_FID_NUM_IMAGES = 12800

# helper classes

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

class Flatten(nn.Layer):
    def forward(self, x):
        return x.reshape([x.shape[0], -1])

class RandomApply(nn.Layer):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class Rezero(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = self.create_parameter([1], default_initializer=nn.initializer.Constant(0.0))
    def forward(self, x):
        return self.fn(x) * self.g

class transposeToFrom(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.transpose([0, 2, 3, 1])
        out, loss = self.fn(x)
        out = out.transpose([0, 3, 1, 2])
        return out, loss

class Blur(nn.Layer):
    def __init__(self):
        super().__init__()
        f = paddle.to_tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f.astype(x.dtype)
        f = f.unsqueeze(0).unsqueeze(0) * f.unsqueeze(0).unsqueeze(2)
        return filter2D(x, f, normalized=True)

# one layer of self-attention and feedforward, for images

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
    Residual(Rezero(nn.Sequential(nn.Conv2D(chan, chan * 2, 1), leaky_relu(), nn.Conv2D(chan * 2, chan, 1))))
])

# helpers

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(value, d):
    return value if exists(value) else d

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def is_empty(t):
    if isinstance(t, paddle.Tensor):
        return t.reshape([-1]).shape[0] == 0 # t.nelement() == 0
    return not exists(t)

def raise_if_nan(t):
    if paddle.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        # TODO: I don't know how to waiting for DataParallel in paddle
        # num_no_syncs = gradient_accumulate_every - 1
        # head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        # tail = [null_context]
        # contexts =  head + tail
        assert gradient_accumulate_every > 1, 'gradient_accumulate_every is not supported for ddp in current version' 
        contexts = [null_context]
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = paddle_grad(outputs=output, inputs=images,
                            grad_outputs=paddle.ones(output.shape),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape([batch_size, -1])
    return weight * ((gradients.norm(2, 1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = paddle.randn(images.shape) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = paddle_grad(outputs=outputs, inputs=styles,
                           grad_outputs=paddle.ones(outputs.shape),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(2).mean(1).sqrt()

def noise(n, latent_dim):
    return paddle.randn([n, latent_dim])

def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]

def mixed_list(n, layers, latent_dim):
    tt = int(paddle.rand([1]).numpy()[0] * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size):
    return paddle.uniform([n, im_size, im_size, 1], min=0., max=1.)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x[:x.shape[0]//max_batch_size*max_batch_size].split(max_batch_size, 0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return paddle.concat(chunked_outputs, 0)

def styles_def_to_tensor(styles_def):
    return paddle.concat([t.unsqueeze(1).expand([-1, n, -1]) for t, n in styles_def if n > 0], 1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.trainable = bool

def slerp(val, low, high):
    low_norm = low / paddle.norm(low, axis=1, keepdim=True)
    high_norm = high / paddle.norm(high, axis=1, keepdim=True)
    omega = paddle.fluid.layers.acos((low_norm * high_norm).sum(1))
    so = paddle.fluid.layers.sin(omega)
    res = (paddle.fluid.layers.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (paddle.fluid.layers.sin(val * omega) / so).unsqueeze(1) * high
    return res

# dataset

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand([3, -1, -1])
        elif channels == 2:
            color = tensor[:1].expand([3, -1, -1])
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = paddle.ones([1, *tensor.shape[1:]])

        return color if not self.transparent else paddle.concat([color, alpha])

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return paddle.vision.transforms.functional.resize(image, min_size)
    return image

class Dataset(io.Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            convert_image_fn, # transforms.Lambda(convert_image_fn),
            partial(resize_to_minimum_size, image_size), # transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            lambda img: np.asarray(img).transpose([2, 0, 1]).astype(np.float32)/255.0, # transforms.ToTensor(), this raise Error
            expand_greyscale(transparent) # transforms.Lambda(expand_greyscale(transparent))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return [self.transform(img)]

# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return paddle.flip(tensor, axis=[3,])

class AugWrapper(nn.Layer):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)

# stylegan2 classes

class EqualLinear(nn.Layer):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = self.create_parameter((in_dim, out_dim))
        if bias:
            self.bias = self.create_parameter((out_dim,), is_bias=True)

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Layer):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, axis=1)
        return self.net(x)

class RGBBlock(nn.Layer):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(

            # TODO: The Op depthwise_conv2d_grad and pad3d_grad doesn't have any grad op for gradient penalty in current paddle version
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),

            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Layer):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = self.create_parameter(
            (out_chan, in_chan, kernel, kernel),
            default_initializer=nn.initializer.KaimingNormal()
        )

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y.unsqueeze(1).unsqueeze(3).unsqueeze(4)
        w2 = self.weight.unsqueeze(0)
        weights = w2 * (w1 + 1)

        if self.demod:
            d = paddle.rsqrt((weights ** 2).sum([2, 3, 4], keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape([1, -1, h, w])

        _, _, *ws = weights.shape
        weights = weights.reshape([b * self.filters, *ws])

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)

        # TODO: The Op depthwise_conv2d_grad doesn't have any grad op for gradient penalty in current paddle version
        # x = F.conv2d(x, weights, padding=padding, groups=b)
        x = x.reshape([b, -1, h, w]) * (y.unsqueeze(2).unsqueeze(3) + 1)
        x = F.conv2d(x, self.weight, padding=padding) * (d[:,:,:,:,0] if self.demod else 1)

        x = x.reshape([-1, self.filters, h, w])
        return x

class GeneratorBlock(nn.Layer):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        # TODO: The Op depthwise_conv2d_grad and pad3d_grad doesn't have any grad op for gradient penalty in current paddle version
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).transpose([0, 3, 2, 1])
        noise2 = self.to_noise2(inoise).transpose([0, 3, 2, 1])

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

class DiscriminatorBlock(nn.Layer):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2D(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2D(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2D(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2D(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Generator(nn.Layer):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.Conv2DTranspose(latent_dim, init_channels, 4, 1, 0, bias_attr=False)
        else:
            self.initial_block = self.create_parameter((1, init_channels, 4, 4))

        self.initial_conv = nn.Conv2D(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.LayerList([])
        attns = []

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None
            attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)

        self._attns = attns
        self.attns = nn.LayerList(filter(lambda l:exists(l), attns))

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(1).unsqueeze(2).unsqueeze(3)
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand([batch_size, -1, -1, -1])

        rgb = None
        styles = styles.transpose([1, 0, 2])
        x = self.initial_conv(x)
        for style, block, attn in zip(styles, self.blocks, self._attns):
            if exists(attn):
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb

class Discriminator(nn.Layer):
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [], transparent = False, fmap_max = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(64) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None
            attn_blocks.append(attn_fn)

            quantize_fn = transposeToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.LayerList(blocks)
        self._attn_blocks = attn_blocks
        self.attn_blocks = nn.LayerList(filter(lambda l:exists(l), attn_blocks))
        self._quantize_blocks = quantize_blocks
        self.quantize_blocks = nn.LayerList(filter(lambda l:exists(l), quantize_blocks))

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2D(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = paddle.zeros([1])
        for (block, attn_block, q_block) in zip(self.blocks, self._attn_blocks, self._quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, _, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss

class StyleGAN2(nn.Layer):
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, cl_reg = False, steps = 1, lr = 1e-4, ttur_mult = 2, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, lr_mlp = 0.1, rank = 0):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, transparent = transparent, fmap_max = fmap_max)

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const)

        self.D_cl = None

        if cl_reg:
            # TODO: kornia package is too heavy for me to convert to paddle version
            assert False, 'Still not support contrastive_learner now'
            from stylegan2_paddle.contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(parameters=generator_params, learning_rate=self.lr, beta1=0.5, beta2=0.9)
        self.D_opt = Adam(parameters=self.D.parameters(), learning_rate=self.lr * ttur_mult, beta1=0.5, beta2=0.9)

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        # startup amp mixed precision
        self.fp16 = fp16
        if fp16:
            # TODO: Paddle AMP is not totally similar to PyTorch. Not to implement it in current version
            assert False, 'fp16 training is not supported in current version.'
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.sublayers():
            if type(m) in {nn.Conv2D, nn.Linear}:
                nn.initializer.KaimingNormal()(m.weight)

        for block in self.G.blocks:
            nn.initializer.Constant(0.0)(block.to_noise1.weight)
            nn.initializer.Constant(0.0)(block.to_noise2.weight)
            nn.initializer.Constant(0.0)(block.to_noise1.bias)
            nn.initializer.Constant(0.0)(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.set_state_dict(self.S.state_dict())
        self.GE.set_state_dict(self.G.state_dict())

    def forward(self, x):
        return x

class Trainer():
    def __init__(self, name, results_dir, models_dir, image_size, network_capacity, transparent = False, batch_size = 4, mixed_prob = 0.9, gradient_accumulate_every=1, lr = 2e-4, lr_mlp = 1., ttur_mult = 2, rel_disc_loss = False, num_workers = None, save_every = 1000, trunc_psi = 0.6, fp16 = False, cl_reg = False, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, aug_prob = 0., aug_types = ['translation', 'cutout'], top_k_training = False, generator_top_k_gamma = 0.99, generator_top_k_frac = 0.5, dataset_aug_prob = 0., calculate_fid_every = None, is_ddp = False, rank = 0, world_size = 1, *args, **kwargs):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
    
    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, image_size = self.image_size, network_capacity = self.network_capacity, transparent = self.transparent, fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, rank = self.rank, *args, **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            paddle.distributed.init_parallel_env()
            self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder, use_shared_memory=True):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        num_workers = default(self.num_workers, num_cores)
        dataloader = io.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), shuffle = True, drop_last = True, use_shared_memory=use_shared_memory)
        self.loader = cycle(dataloader)

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = paddle.to_tensor(0.)
        total_gen_loss = paddle.to_tensor(0.)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob   = self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0

        # TODO: The Op depthwise_conv2d_grad and pad3d_grad doesn't have any grad op for gradient penalty in current paddle version
        # apply_path_penalty = self.steps > 5000 and self.steps % 32 == 0
        apply_path_penalty = False

        apply_cl_reg_to_generated = self.steps > 20000

        S = self.GAN.S if not self.is_ddp else self.S_ddp
        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        if exists(self.GAN.D_cl):
            self.GAN.D_opt.clear_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                    style = get_latents_fn(batch_size, num_layers, latent_dim)
                    noise = image_noise(batch_size, image_size)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(w_styles, noise)
                    self.GAN.D_cl(generated_images.astype(generated_images.dtype).detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch = next(self.loader)[0]
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.astype(loss.dtype).detach().numpy()[0]
            backwards(loss, self.GAN.D_opt, loss_id = 0)

            self.GAN.D_opt.step()

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.clear_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, S, G]):
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = G(w_styles, noise)
            fake_output, fake_q_loss = D_aug(generated_images.astype(generated_images.dtype).detach(), detach = True, **aug_kwargs)

            image_batch = next(self.loader)[0]
            image_batch.stop_gradient = False
            real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            if self.rel_disc_loss:
                real_output_loss = real_output_loss - fake_output.mean()
                fake_output_loss = fake_output_loss - real_output.mean()

            divergence = (F.relu(1 + real_output_loss) + F.relu(1 - fake_output_loss)).mean()
            disc_loss = divergence

            if self.has_fq:
                quantize_loss = (fake_q_loss + real_q_loss).mean()
                self.q_loss = float(quantize_loss.detach().numpy()[0])

                disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.astype(gp.dtype).detach().numpy()[0]
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            # TODO: Not support in Paddle # disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id = 1)

            total_disc_loss += divergence.astype(divergence.dtype).detach().numpy()[0] / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.clear_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[S, G, D_aug]):
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = G(w_styles, noise)
            fake_output, _ = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            loss = fake_output_loss.mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not paddle.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            # TODO: Not support in Paddle # gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, loss_id = 2)

            total_gen_loss += loss.detach().numpy()[0] / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(paddle.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % EVALUATE_EVERY == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(floor(self.steps / EVALUATE_EVERY))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(CALC_FID_NUM_IMAGES / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1
        self.av = None

    @paddle.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 8, trunc = 1.0):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)
        utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
        utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.shape[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.tile(repeat_idx)
            order_index = paddle.to_tensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]).astype(np.int64))
            return paddle.index_select(a, axis=dim, index=order_index)

        nn = noise(num_rows, latent_dim)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.tile([num_rows, 1])

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @paddle.no_grad()
    def calculate_fid(self, num_batches):
        paddle.cuda.empty_cache()

        real_path = str(self.results_dir / self.name / 'fid_real') + '/'
        fake_path = str(self.results_dir / self.name / 'fid_fake') + '/'

        # remove any existing files used for fid calculation and recreate directories
        rmtree(real_path, ignore_errors=True)
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(real_path)
        os.makedirs(fake_path)

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
            real_batch = next(self.loader)[0]
            for k in range(real_batch.size(0)):
                utils.save_image(real_batch[k, :, :, :], real_path + '{}.png'.format(k + batch_num * self.batch_size))

        # generate a bunch of fake images in results / name / fid_fake
        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim)
            n = image_noise(self.batch_size, image_size)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)

            for j in range(generated_images.size(0)):
                utils.save_image(generated_images[j, :, :, :], str(Path(fake_path) / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([real_path, fake_path], 256, True, 2048)

    @paddle.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        latent_dim = G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, S, z).numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)
            
        w_space = []
        for tensor, num_layers in style:
            tmp = S(tensor)
            av_paddle = paddle.to_tensor(self.av)
            tmp = trunc_psi * (tmp - av_paddle) + av_paddle
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clip(0., 1.)

    @paddle.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim)
        latents_high = noise(num_rows ** 2, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        ratios = paddle.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            images_grid = utils.make_grid(generated_images, nrow = num_rows)
            pil_image = Image.fromarray((images_grid*255+0.5).clip(0, 255).transpose([1, 2, 0]).astype(paddle.uint8).numpy())
            
            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def model_name(self, num, ext='pkl'):
        return str(self.models_dir / self.name / f'model_{num}.{ext}')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(f'./models/{self.name}', True)
        rmtree(f'./results/{self.name}', True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'version': __version__
        }

        paddle.save(self.GAN.state_dict(), self.model_name(num, 'pdparams'))
        paddle.save(save_data, self.model_name(num))
        if self.GAN.fp16:
            paddle.save(amp.state_dict(), self.model_name(num, 'amp.pdparams'))

        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pkl')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = paddle.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            load_GAN_data = paddle.load(self.model_name(name, 'pdparams'))
            self.GAN.set_state_dict(load_GAN_data)
        except:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            exit()

        if self.GAN.fp16 and 'amp' in os.path.exists(self.model_name(name, 'amp.pdparams')):
            load_amp_data = paddle.load(self.model_name(name, 'amp.pdparams'))
            amp.set_state_dict(load_amp_data)
