#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import os.path

from setuptools import find_packages
from setuptools import setup

def get_version():
    g = {}
    exec(open(os.path.join("awkward", "version.py")).read(), g)
    return g["__version__"]

def get_description():
    description = open("README.rst", "rb").read().decode("utf8", "ignore")
    start = description.index(".. inclusion-marker-1-5-do-not-remove")
    stop = description.index(".. inclusion-marker-3-do-not-remove")

    before = """.. image:: https://raw.githubusercontent.com/scikit-hep/awkward-array/master/docs/source/logo-300px.png
   :alt: awkward-array
   :target: https://github.com/scikit-hep/awkward-array

|

"""

    after = """

Tutorial
========

See the `project homepage <https://github.com/scikit-hep/awkward-array>`__ for a `tutorial <https://github.com/scikit-hep/awkward-array#tutorial>`__.

Interactive tutorial
====================

Run `this tutorial <https://mybinder.org/v2/gh/scikit-hep/awkward-array/master?filepath=binder%2Ftutorial.ipynb>`__ on Binder.

Reference documentation
=======================

"""

    return before + description[start:stop].strip() # + after

setup(name = "awkward",
      version = get_version(),
      packages = find_packages(exclude = ["tests"]),
      scripts = [],
      description = "Manipulate arrays of complex data structures as easily as Numpy.",
      long_description = get_description(),
      author = "Jim Pivarski (IRIS-HEP)",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski (IRIS-HEP)",
      maintainer_email = "pivarski@princeton.edu",
      url = "https://github.com/scikit-hep/awkward-array",
      download_url = "https://github.com/scikit-hep/awkward-array/releases",
      license = "BSD 3-clause",
      test_suite = "tests",
      install_requires = ["numpy>=1.13.1"],
      setup_requires = ["pytest-runner"],
      tests_require = ["pytest", "pandas"],
      classifiers = [
          # "Development Status :: 1 - Planning",
          # "Development Status :: 2 - Pre-Alpha",
          # "Development Status :: 3 - Alpha",
          # "Development Status :: 4 - Beta",
          "Development Status :: 5 - Production/Stable",
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
