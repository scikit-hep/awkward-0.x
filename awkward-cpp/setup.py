#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import os.path

import pybind11

from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages
from setuptools import setup

def get_version():
    g = {}
    exec(open(os.path.join("..", "awkward", "version.py")).read(), g)
    return g["__version__"]

setup(name = "awkward-cpp",
      version = get_version(),
      packages = find_packages(exclude = ["tests"]),
      scripts = [],
      description = "Connect awkward-arrays to C++ using pybind11.",
      long_description = "",
      author = "Charles Escott",
      author_email = "charlescescott@gmail.com",
      maintainer = "Jim Pivarski (IRIS-HEP)",
      maintainer_email = "pivarski@princeton.edu",
      url = "https://github.com/scikit-hep/awkward-array",
      download_url = "https://github.com/scikit-hep/awkward-array/releases",
      license = "BSD 3-clause",
      test_suite = "tests",
      install_requires = ["awkward==" + get_version(), "pybind11"],
      setup_requires = ["pytest-runner"],
      tests_require = ["pytest"],
      ext_modules = [Extension("awkward.cpp.array._jagged",
                               ["awkward/cpp/array/_jagged.cpp"],
                               include_dirs=[pybind11.get_include(False),
                                             pybind11.get_include(True)],
                               language="c++")
                     ],
      classifiers = [
          "Development Status :: 1 - Planning",
          # "Development Status :: 2 - Pre-Alpha",
          # "Development Status :: 3 - Alpha",
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
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development",
          "Topic :: Utilities",
          ],
      platforms = "Any",
      )
