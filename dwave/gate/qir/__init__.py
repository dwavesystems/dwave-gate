# Copyright 2023 D-Wave Systems Inc.
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

"""Quantum Intermediate Representation (QIR) module.

Contains the QIR compiler, loader and related files and functions.
"""

from typing import Tuple

try:
    import pyqir  # noqa: F401
except ImportError:  # pragma: no cover
    pyqir_installed = False
else:
    # pyqir>=0.8 required
    from importlib.metadata import version

    def parse(version: str) -> Tuple[str, str, str]:
        "Parse a MAJOR.MINOR.PATCH version string into a tuple."
        return tuple(version.split("."))[:3]

    pyqir_installed = parse(version("pyqir")) >= parse("0.8.0")

if not pyqir_installed:  # pragma: no cover
    raise ImportError("PyQIR not installed.")

from dwave.gate.qir.compiler import *
from dwave.gate.qir.loader import *
