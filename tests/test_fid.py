import os
import numpy as np
from PIL import Image

from stylegan2_paddle.fid import fid_score

result_dir = 'test_fid_results'
os.makedirs(result_dir, exist_ok=True)

real_path = os.path.join(result_dir, 'real_imgs')
os.makedirs(real_path, exist_ok=True)
for i in range(4):
    img = np.random.randint(0,256,size=[256,256,3]).astype('uint8')
    img = Image.fromarray(img)
    img.save(os.path.join(real_path, str(i).zfill(3)+'.png'))

fake_path = os.path.join(result_dir, 'fake_imgs')
os.makedirs(fake_path, exist_ok=True)
for i in range(4):
    img = np.random.randint(0,256,size=[256,256,3]).astype('uint8')
    img = Image.fromarray(img)
    img.save(os.path.join(fake_path, str(i).zfill(3)+'.png'))

out = fid_score.calculate_fid_given_paths([real_path, fake_path], 256, True, 2048)
print('fid score:', out)