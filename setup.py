# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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
        name='dwave.gate.simulator.simulator',
        sources=['dwave/gate/simulator/simulator.pyx'],
        include_dirs=["dwave/gate/simulator/"],
        language='c++',
    ),
    Extension(
        name='dwave.gate.registers.cyregister',
        sources=['dwave/gate/registers/cyregister.pyx'],
        include_dirs=["dwave/gate/registers/"],
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
