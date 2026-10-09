"""Build the Cython extension for fast-flow."""

import platform

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# Optimization + OpenMP.  The core loops are written GIL-free
# (Cython prange over raw buffers), so OpenMP threads them across
# cores.  Apple clang does not bundle OpenMP: on macOS it needs
# `brew install libomp` and the -Xpreprocessor spelling.
if platform.system() == "Darwin":
    compile_args = ["-O3", "-Xpreprocessor", "-fopenmp"]
    if platform.machine() == "arm64":
        compile_args.append("-mcpu=native")
    link_args = ["-lomp"]
else:
    compile_args = ["-O3", "-march=native", "-fopenmp"]
    link_args = ["-fopenmp"]

def _extension(name, source):
    return Extension(
        name,
        sources=[source],
        include_dirs=[
            np.get_include(),
            "/opt/homebrew/opt/libomp/include"
        ],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        library_dirs=[
            "/opt/homebrew/opt/libomp/lib"
        ]
    )


extensions = [
    _extension("fast_flow.core.flow",
               "src/fast_flow/core/flow.pyx"),
    _extension("fast_flow.core.handlers.forcehandler",
               "src/fast_flow/core/handlers/forcehandler.pyx"),
    _extension("fast_flow.core.electrostaticforcehandler",
               "src/fast_flow/core/electrostaticforcehandler.pyx"),
]

setup(ext_modules=cythonize(extensions, language_level=3))
