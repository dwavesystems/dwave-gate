# Copyright 2026 D-Wave
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

__version__ = "0.4.0"

# The user guide refers to these as dwave.gate.Result and
# dwave.gate.YieldHandling, so re-export them from the package root. They remain
# importable from dwave.gate.results, which is where they are defined.
from dwave.gate.results import Result, YieldHandling

__all__ = [
    "Result",
    "YieldHandling",
]
