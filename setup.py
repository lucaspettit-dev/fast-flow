"""Build the Cython extension for fast-flow."""

import platform

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# Optimization + pthreads (perf/native-c-threadpool): the
# solver's computation is native C (solver_c.c) on a
# persistent pthread pool sized to the online CPUs.
# -pthread for the pool (both gcc and clang accept it).
if platform.system() == "Darwin":
    compile_args = ["-O3", "-pthread"]
    if platform.machine() == "arm64":
        compile_args.append("-mcpu=native")
    link_args = ["-pthread"]
else:
    compile_args = ["-O3", "-march=native", "-pthread"]
    link_args = ["-pthread"]

def _extension(name, source, extra_sources=None):
    return Extension(
        name,
        sources=[source] + (extra_sources or []),
        include_dirs=[
            np.get_include(),
            "src/fast_flow/core",
            "src/fast_flow/core/handlers",
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
               "src/fast_flow/core/flow.pyx",
               extra_sources=["src/fast_flow/core/native/solver_c.c"]),
    _extension("fast_flow.core.handlers.forcehandler",
               "src/fast_flow/core/handlers/forcehandler.pyx"),
    _extension("fast_flow.core.electrostaticforcehandler",
               "src/fast_flow/core/electrostaticforcehandler.pyx",
               extra_sources=["src/fast_flow/core/native/solver_c.c",
                              "src/fast_flow/core/native/handler_c.c"]),
    _extension("fast_flow.core.renderer",
               "src/fast_flow/core/renderer.pyx"),
]

setup(ext_modules=cythonize(extensions, language_level=3))
