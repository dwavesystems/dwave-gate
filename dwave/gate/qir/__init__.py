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

try:
    import pyqir  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("PyQIR required for using the QIR compiler") from e


from dwave.gate.qir.compiler import *
from dwave.gate.qir.loader import *
