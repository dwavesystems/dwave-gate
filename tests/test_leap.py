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

"""Unit tests for LeapQCDLSimulator in leap.py, with the cloud API mocked."""

import concurrent.futures
import unittest.mock

import orjson
import pytest

from dwave.cloud.client import Client
from dwave.cloud.computation import Future
from dwave.cloud.solver import StructuredSolver, QCDLSolver
from dwave.cloud.testing.mocks import qpu_clique_solver_data, qcdl_solver_data

from dwave.gate.qcdl import LeapQCDLSimulator, qcdl
from dwave.gate.results import Result


class mock_client_factory:
    """Stand-in for :class:`dwave.cloud.client.Client` that serves a static
    list of mock solvers, so solver selection runs client-side, unmodified.
    """

    @classmethod
    def from_config(cls, **kwargs):
        # keep instantiation local, so we can later mock BaseUnstructuredSolver
        qcdl_sim_v1 = QCDLSolver(client=None, data=qcdl_solver_data(
            name='qcdl_sim_v1', category='software-gate', version='0.1'))
        qcdl_sim_v2 = QCDLSolver(client=None, data=qcdl_solver_data(
            name='qcdl_sim_v2', category='software-gate', version='0.2'))
        qcdl_qpu = QCDLSolver(client=None, data=qcdl_solver_data(
            name='qcdl_qpu', category='qpu-gate'))
        ising_qpu = StructuredSolver(client=None, data=qpu_clique_solver_data(4))
        kwargs.setdefault('endpoint', 'mock')
        kwargs.setdefault('token', 'mock')
        client = Client(**kwargs)
        client._fetch_solvers = \
            lambda **kwargs: [ising_qpu, qcdl_sim_v1, qcdl_sim_v2, qcdl_qpu]
        return client


class TestLeapQCDLSimulator:

    @unittest.mock.patch('dwave.gate.qcdl.leap.Client', mock_client_factory)
    def test_solver_selection(self):
        simulator = LeapQCDLSimulator()
        try:
            properties = simulator.solver.properties
            assert 'qcdl' in properties.get('supported_problem_types', [])
            assert properties.get('category') == 'software-gate'
            # the latest of the two matching solvers is preferred
            assert properties.get('version') == '0.2'
        finally:
            simulator.close()

    @unittest.mock.patch('dwave.gate.qcdl.leap.Client', mock_client_factory)
    def test_solver_selection_override(self):
        # explicit solver selection takes precedence over the defaults
        simulator = LeapQCDLSimulator(solver=dict(name='qcdl_sim_v1'))
        try:
            assert simulator.solver.identity.name == 'qcdl_sim_v1'
        finally:
            simulator.close()

    def test_invalid_defaults(self):
        with pytest.raises(TypeError):
            LeapQCDLSimulator(defaults='not-a-mapping')

    @unittest.mock.patch('dwave.gate.qcdl.leap.Client', mock_client_factory)
    @unittest.mock.patch('dwave.cloud.solver.BaseUnstructuredSolver.sample_problem')
    @unittest.mock.patch('dwave.cloud.solver.QCDLSolver.decode_response')
    def test_run(self, decode_response, base_sample_problem):
        simulator = LeapQCDLSimulator()

        # create a program
        @qcdl()
        def circuit(q0):
            q0.x()
            q0.measure()

        program = circuit()

        num_qubits = 1
        num_shots = 100

        mock_problem_id = '321'
        mock_answer = dict(num_shots=num_shots, num_qubits=num_qubits, simulated_qcdl='input')

        # note: instead of simply mocking `simulator.solver`, we mock a set of
        # solver methods minimally required to fully test `QCDLSolver.sample_qcdl`

        base_sample_problem.return_value = Future(
            solver=simulator.solver, id_=mock_problem_id)
        base_sample_problem.return_value._set_message({"answer": {}})

        def mock_decode_response(msg, answer_data):
            # write the serialized answer to the "received" answer_data
            answer_data.write(orjson.dumps(mock_answer))
            answer_data.seek(0)
            return {'problem_type': 'qcdl', 'answer': answer_data}

        decode_response.side_effect = mock_decode_response

        try:
            result = simulator.run(program, shots=num_shots)

            # low-level sample_problem called with our program and params
            base_sample_problem.assert_called_with(program, label=None, shots=num_shots)

            # decoded answer returned in a Future[Result]
            assert isinstance(result, concurrent.futures.Future)
            answer = result.result(timeout=10)
            assert isinstance(answer, Result)
            assert answer.num_shots == mock_answer['num_shots']
            assert answer.num_qubits == mock_answer['num_qubits']
            assert answer.simulated_qcdl == mock_answer['simulated_qcdl']
        finally:
            simulator.close()

    @unittest.mock.patch('dwave.gate.qcdl.leap.Client', mock_client_factory)
    def test_close(self):
        simulator = LeapQCDLSimulator()
        with unittest.mock.patch.object(
                simulator.client, 'close', wraps=simulator.client.close) as client_close:
            simulator.close()
        client_close.assert_called_once()
