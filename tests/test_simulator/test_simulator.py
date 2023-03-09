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

import numpy as np
import pytest

import dwave.gate.operations as ops
from dwave.gate.circuit import Circuit, CircuitError
from dwave.gate.operations.operations import __all__ as all_ops
from dwave.gate.simulator.simulator import simulate


def test_simulate_sv_empty_circuit(empty_circuit):
    simulate(empty_circuit)
    assert empty_circuit.state is None


def test_simulate_sv_no_ops_one_qubit():
    circuit = Circuit(1)
    simulate(circuit)
    assert np.all(circuit.state == np.array([1, 0]))


def test_simulate_sv_no_ops_two_qubit():
    circuit = Circuit(2)
    simulate(circuit)
    assert np.all(circuit.state == np.array([1, 0, 0, 0]))


def test_simulate_dm_empty_circuit(empty_circuit):
    simulate(empty_circuit, mixed_state=True)
    assert empty_circuit.state is None


def test_simulate_dm_no_ops_one_qubit():
    circuit = Circuit(1)
    simulate(circuit, mixed_state=True)
    with pytest.raises(CircuitError, match="State is mixed."):
        circuit.state
    assert np.all(circuit.density_matrix == np.array([[1, 0], [0, 0]]))


def test_simulate_dm_no_ops_two_qubit():
    circuit = Circuit(2)
    simulate(circuit, mixed_state=True)
    pure_zero = np.zeros((4, 4))
    pure_zero[0, 0] = 1
    with pytest.raises(CircuitError, match="State is mixed."):
        circuit.state
    assert np.all(circuit.density_matrix == pure_zero)


def test_simulate_sv_not():
    circuit = Circuit(1)
    with circuit.context as regs:
        ops.X(regs.q[0])
    simulate(circuit)
    assert np.all(circuit.state == np.array([0, 1]))


def test_simulate_dm_not():
    circuit = Circuit(1)
    with circuit.context as regs:
        ops.X(regs.q[0])
    simulate(circuit, mixed_state=True)
    assert np.all(circuit.density_matrix == np.array([[0, 0], [0, 1]]))


def test_simulate_sv_cnot():
    circuit = Circuit(2)
    with circuit.context as regs:
        ops.X(regs.q[0])
        ops.CNOT(regs.q[0], regs.q[1])
    simulate(circuit)
    # should be |11>
    assert np.all(circuit.state == np.array([0, 0, 0, 1]))


def test_simulate_dm_cnot():
    circuit = Circuit(2)
    with circuit.context as regs:
        ops.X(regs.q[0])
        ops.CNOT(regs.q[0], regs.q[1])
    simulate(circuit, mixed_state=True)
    # should be |11>
    pure_11 = np.zeros((4, 4))
    pure_11[3, 3] = 1
    assert np.all(circuit.density_matrix == pure_11)


def test_simulate_sv_big_endian():
    circuit = Circuit(2)
    with circuit.context as regs:
        ops.X(regs.q[1])

    simulate(circuit, little_endian=False)
    assert np.all(circuit.state == np.array([0, 1, 0, 0]))


def test_simulate_sv_little_endian():
    circuit = Circuit(2)
    with circuit.context as regs:
        ops.X(regs.q[1])

    simulate(circuit, little_endian=True)
    assert np.all(circuit.state == np.array([0, 0, 1, 0]))


def test_simulate_sv_swap():
    circuit = Circuit(2)
    with circuit.context as regs:
        ops.X(regs.q[0])
        ops.Hadamard(regs.q[0])
        ops.Hadamard(regs.q[1])
        ops.SWAP([regs.q[0], regs.q[1]])

    simulate(circuit, little_endian=False)

    assert np.all(np.isclose(circuit.state, 0.5 * np.array([1, -1, 1, -1])))


def test_simulate_dm_swap():
    circuit = Circuit(2)
    with circuit.context as regs:
        ops.X(regs.q[0])
        ops.Hadamard(regs.q[0])
        ops.Hadamard(regs.q[1])
        ops.SWAP([regs.q[0], regs.q[1]])

    simulate(circuit, little_endian=False)

    assert np.all(np.isclose(circuit.state, 0.5 * np.array([1, -1, 1, -1])))


def test_simulate_sv_ccx():
    circuit = Circuit(3)
    with circuit.context as regs:
        ops.X(regs.q[0])
        ops.X(regs.q[2])
        ops.CCX([regs.q[0], regs.q[2], regs.q[1]])

    simulate(circuit)

    # CCX should have taken |101> to |111>
    assert np.all(circuit.state == np.array([0, 0, 0, 0, 0, 0, 0, 1]))


def test_simulate_dm_ccx():
    circuit = Circuit(3)
    with circuit.context as regs:
        ops.X(regs.q[0])
        ops.X(regs.q[2])
        ops.CCX([regs.q[0], regs.q[2], regs.q[1]])

    simulate(circuit, mixed_state=True)

    # CCX should have taken |101> to |111>
    pure_111 = np.zeros((8, 8))
    pure_111[7, 7] = 1

    assert np.all(circuit.density_matrix == pure_111)


def test_simulate_sv_cswap():
    circuit = Circuit(3)
    with circuit.context as regs:
        ops.X(regs.q[0])
        ops.X(regs.q[1])
        ops.CSWAP([regs.q[1], regs.q[0], regs.q[2]])

    simulate(circuit, little_endian=False)

    # CSWAP should have mapped |011> to |110>
    assert np.all(circuit.state == np.array([0, 0, 0, 1, 0, 0, 0, 0]))


def test_simulate_dm_cswap():
    circuit = Circuit(3)
    with circuit.context as regs:
        ops.X(regs.q[0])
        ops.X(regs.q[1])
        ops.CSWAP([regs.q[1], regs.q[0], regs.q[2]])

    simulate(circuit, little_endian=False, mixed_state=True)

    # CSWAP should have mapped |011> to |110>
    pure_110 = np.zeros((8, 8))
    pure_110[3, 3] = 1

    assert np.all(circuit.density_matrix == pure_110)


@pytest.mark.parametrize("op", [getattr(ops, name) for name in all_ops])
@pytest.mark.parametrize("little_endian", [False, True])
@pytest.mark.parametrize("mixed_state", [False, True])
def test_simulate_all_gates(op, little_endian, mixed_state):
    circuit = Circuit(op.num_qubits)
    kwargs = {}
    with circuit.context as regs:
        if issubclass(op, ops.ParametricOperation):
            # TODO random parameters?
            kwargs["parameters"] = [0 for i in range(op.num_parameters)]
        kwargs["qubits"] = [regs.q[i] for i in range(op.num_qubits)]

        op(**kwargs)

    simulate(circuit, little_endian=little_endian, mixed_state=mixed_state)

    if mixed_state:
        assert 1 == pytest.approx(np.trace(circuit.density_matrix))
    else:
        assert 1 == pytest.approx(np.sum(circuit.state * circuit.state.conj()))
