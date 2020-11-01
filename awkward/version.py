#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import re

__version__ = "0.14.0"
version = __version__
version_info = tuple(re.split(r"[-\.]", __version__))

del re
