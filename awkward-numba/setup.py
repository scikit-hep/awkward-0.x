#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import os.path

from setuptools import find_packages
from setuptools import setup

def get_version():
    g = {}
    exec(open(os.path.join("..", "awkward", "version.py")).read(), g)
    return g["__version__"]

setup(name = "awkward-numba",
      version = get_version(),
      packages = find_packages(exclude = ["tests"]),
      scripts = [],
      description = "Allows awkward arrays to be used in Numba-compiled code and optimizes awkward methods with JIT compilation.",
      long_description = """.. image:: https://raw.githubusercontent.com/scikit-hep/awkward-array/master/docs/source/logo-300px.png
   :alt: awkward-array
   :target: https://github.com/scikit-hep/awkward-array

|

Allows awkward arrays to be used in Numba-compiled code and optimizes awkward methods with JIT compilation.

See tests/study_numba_speed.py for an example.

Note: be sure to ``import awkward.numba`` before attempting to use an awkward array in a Numba routine.

Status: only JaggedArrays have been implemented.
""",
      author = "Jim Pivarski (IRIS-HEP)",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski (IRIS-HEP)",
      maintainer_email = "pivarski@princeton.edu",
      url = "https://github.com/scikit-hep/awkward-array",
      download_url = "https://github.com/scikit-hep/awkward-array/releases",
      license = "BSD 3-clause",
      test_suite = "tests",
      install_requires = ["awkward==" + get_version(), "numba"],
      setup_requires = ["pytest-runner"],
      tests_require = ["pytest"],
      classifiers = [
          # "Development Status :: 1 - Planning",
          # "Development Status :: 2 - Pre-Alpha",
          "Development Status :: 3 - Alpha",
          # "Development Status :: 4 - Beta",
          # "Development Status :: 5 - Production/Stable",
          # "Development Status :: 6 - Mature",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Operating System :: MacOS",
          "Operating System :: POSIX",
          "Operating System :: Unix",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development",
          "Topic :: Utilities",
          ],
      platforms = "Any",
      )
