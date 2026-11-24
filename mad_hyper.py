"""Compatibility module: MAD-HYPE functions (faithful port of mad_hyper.R), see tirtl.madhyper."""
import os
import sys

try:
    import tirtl  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from tirtl.madhyper import *  # noqa: F401,F403,E402
from tirtl.madhyper import _dmultinom as _dmultinom_log  # noqa: F401,E402
