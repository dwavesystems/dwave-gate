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

"""Provide access to QCDL solvers in Leap."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import orjson
from dwave.cloud.client import Client

from dwave.gate.qcdl.qcdl_models import QCDLProgram
from dwave.gate.results import Result

__all__ = ['LeapQCDLSimulator']


class LeapQCDLSimulator:
    r"""Submits QCDL programs to the Dual-Rail simulator in the Leap service.
    """

    @property
    def default_solver(self) -> dict[str, str]:
        """dict: Features used to select the latest accessible QCDL software solver."""
        return dict(supported_problem_types__contains='qcdl',
                    category='software-gate',
                    order_by='-properties.version')

    @property
    def properties(self) -> dict[str, Any]:
        """Solver properties as returned by a :term:`SAPI` query.

        Solver properties are dependent on the selected solver and subject to
        change; for example, new features may add properties.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self) -> dict[str, list[str]]:
        """Supported parameters.

        Keys of the returned dict are keyword parameters as returned by a
        :term:`SAPI` query.

        Solver parameters are dependent on the selected solver and subject to
        change; for example, new features may add parameters.
        """
        try:
            return self._parameters
        except AttributeError:
            parameters = {param: ['parameters']
                          for param in self.properties['parameters']}
            parameters.update(label=[])
            self._parameters = parameters
            return parameters

    def __init__(self, **config):
        """Initialize the simulator client instance.

        Args:
            **config:
                :class:`~dwave.cloud.client.Client` configuration options,
                including the :term:`solver` selection.

        """
        # default to short-lived session to prevent resets on slow uploads
        config.setdefault('connection_close', True)

        # prefer the latest QCDL solver available, but allow for an easy
        # override on any config level above the defaults (file/env/kwarg)
        defaults = config.setdefault('defaults', {})
        if not isinstance(defaults, Mapping):
            raise TypeError("mapping expected for 'defaults'")
        defaults.update(solver=self.default_solver)

        self.client = Client.from_config(**config)
        self.solver = self.client.get_solver()

        # check user-specified solver conforms to our requirements
        if self.properties.get('category') != 'software-gate':
            raise ValueError("selected solver is not a gate-model simulator.")
        if 'qcdl' not in self.solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'qcdl' problem type.")

        self._executor = ThreadPoolExecutor()

    def close(self) -> None:
        """Close the underlying cloud client to release system resources such as
        threads.

        The method blocks for all the currently scheduled work (sampling
        requests) to finish.

        See also:
            :meth:`~dwave.cloud.client.Client.close`.
        """
        self.client.close()
        self._executor.shutdown(wait=True)

    def run(self, qcdl: QCDLProgram | Mapping[str, Any], **params) -> Future[Result]:
        """Run the :term:`QCDL` program using the selected Leap simulator,
        and return the :class:`~dwave.gate.results.Result` in a
        class:`~concurrent.futures.Future`.

        Args:
            qcdl:
                The QCDL circuit to upload and simulate.
            **params:
                Job parameters accepted by the simulator (solver).

        Returns:
            Simulation results.

        """
        response = self.solver.sample_qcdl(qcdl, **params)

        def decode():
            answer = orjson.loads(response.answer_data.read())
            return Result(**answer)

        result = self._executor.submit(decode)

        return result
