#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import re

__version__ = "0.15.3"
version = __version__
version_info = tuple(re.split(r"[-\.]", __version__))

del re
