#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import os.path
import pybind11
import sys
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages
from setuptools import setup

def get_version():
    g = {}
    exec(open(os.path.join("..", "awkward", "version.py")).read(), g)
    return g["__version__"]

#################################################################################
# The following code is copy-pasted from pybind/python_example/setup.py on github:

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'awkward.cpp.array.array_impl',
        [os.path.join("awkward", "cpp", "array", "array_impl.cpp")],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
]

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

############################# end copy-pasted code ##############################

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
      install_requires = ["awkward==" + get_version(), "pybind11>=2.2"],
      setup_requires = ["pytest-runner"],
      tests_require = ["pytest"],
      ext_modules = ext_modules,
      cmdclass = {"build_ext": BuildExt},
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
