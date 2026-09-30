"""Build the Cython extension for fast-flow."""

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "fast_flow._core",
        sources=["src/fast_flow/_core.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(ext_modules=cythonize(extensions, language_level=3))
