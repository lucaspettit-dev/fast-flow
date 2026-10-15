"""Build the Cython extension for fast-flow."""

import platform

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# Optimization + OpenMP (Idea 3): the hot loops are
# row-parallel (Jacobi / per-cell, rows independent) and
# run under prange.  The earlier prange was removed in
# 826e795 because it parallelized over the sweep index
# (a race: threads flipped the shared buffer index);
# row-parallelism has no shared writes.
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
    _extension("fast_flow.core.renderer",
               "src/fast_flow/core/renderer.pyx"),
]

setup(ext_modules=cythonize(extensions, language_level=3))
