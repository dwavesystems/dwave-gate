# Confidential & Proprietary Information: D-Wave Systems Inc.
import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import numpy
from Cython.Build import cythonize


extra_compile_args = {
    'msvc': ['/std:c++14', "/openmp"],
    'unix': ['-std=c++11', "-fopenmp", "-Ofast", "-ffast-math",
             "-march=native"],
}

extra_link_args = {
    'msvc': ["/openmp"],
    'unix': ["-fopenmp"],
}


class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type

        compile_args = extra_compile_args[compiler]
        for ext in self.extensions:
            ext.extra_compile_args = compile_args

        link_args = extra_link_args[compiler]
        for ext in self.extensions:
            ext.extra_link_args = link_args

        build_ext.build_extensions(self)


extensions = [
    Extension(
        name='dwgms.simulator.simulator',
        sources=['dwgms/simulator/simulator.pyx'],
        include_dirs=["dwgms/simulator/"],
        language='c++',
    ),
]

setup(
    name="dwave-gate",
    install_requires=[
        "numpy",
    ],
    ext_modules=cythonize(
        extensions, annotate=bool(os.getenv("CYTHON_ANNOTATE", False))
    ),
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': build_ext_compiler_check},
    zip_safe=False,
)
