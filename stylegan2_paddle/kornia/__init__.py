# Make sure that kornia is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)
import sys
if sys.version_info < (3, 6, 0):
    raise RuntimeError("Kornia requires Python 3.6.0 or later")

from stylegan2_paddle.kornia.filters import (
    filter2D,
)