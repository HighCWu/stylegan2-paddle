import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['stylegan2_paddle']
from version import __version__

setup(
  name = 'stylegan2_paddle',
  packages = find_packages(),
  entry_points={
      'console_scripts': [
          'stylegan2_paddle = stylegan2_paddle.cli:main',
      ],
  },
  version = __version__,
  license='GPLv3+',
  description = 'StyleGan2 in PaddlePaddle',
  author = 'Hecong Wu',
  author_email = 'hecongw@gmail.com',
  url = 'https://github.com/HighCWu/stylegan2-paddle',
  download_url = 'https://github.com/HighCWu/stylegan2-paddle/archive/v_036.tar.gz',
  keywords = ['generative adversarial networks', 'artificial intelligence'],
  install_requires=[
      'fire',
      'numpy',
      'retry',
      'tqdm',
      'paddlepaddle-gpu', # paddlepaddle-gpu>=2.0.0
      'pillow'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)