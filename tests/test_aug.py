import os
import paddle
import numpy as np
from PIL import Image

from stylegan2_paddle.diff_augment import DiffAugment

result_dir = 'test_aug_results'
os.makedirs(result_dir, exist_ok=True)

batch_size = 4

imgs = paddle.zeros([batch_size, 3, 128, 128]) + 0.5
imgs = DiffAugment(imgs, ['color'])
imgs_dir = os.path.join(result_dir, 'imgs1')
os.makedirs(imgs_dir, exist_ok=True)
for i, img in enumerate(imgs):
    img = np.uint8(img.numpy().transpose([1,2,0]) * 255)
    img = Image.fromarray(img)
    img.save(os.path.join(imgs_dir, str(i).zfill(3)+'.png'))


imgs = paddle.rand([batch_size, 3, 128, 128])
imgs = DiffAugment(imgs, ['translation'])
imgs_dir = os.path.join(result_dir, 'imgs2')
os.makedirs(imgs_dir, exist_ok=True)
for i, img in enumerate(imgs):
    img = np.uint8(img.numpy().transpose([1,2,0]) * 255)
    img = Image.fromarray(img)
    img.save(os.path.join(imgs_dir, str(i).zfill(3)+'.png'))


imgs = paddle.rand([batch_size, 3, 128, 128])
imgs = DiffAugment(imgs, ['cutout'])
imgs_dir = os.path.join(result_dir, 'imgs3')
os.makedirs(imgs_dir, exist_ok=True)
for i, img in enumerate(imgs):
    img = np.uint8(img.numpy().transpose([1,2,0]) * 255)
    img = Image.fromarray(img)
    img.save(os.path.join(imgs_dir, str(i).zfill(3)+'.png'))

