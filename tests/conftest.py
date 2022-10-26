# Confidential & Proprietary Information: D-Wave Systems Inc.
import pytest

import dwave.gate.operations as ops
from dwave.gate.circuit import Circuit
from dwave.gate.operations.base import ControlledOperation, Operation, ParametricOperation


@pytest.fixture(scope="function")
def two_qubit_circuit():
    """Circuit with two qubits and three operations."""
    circuit = Circuit(2)

    with circuit.context as q:
        ops.Hadamard(q[0])
        ops.CNOT(q[0], q[1])
        ops.Hadamard(q[1])

    return circuit


@pytest.fixture(scope="function")
def two_bit_circuit():
    """Circuit with two bits and one qubit and a single operation."""
    circuit = Circuit(1, 2)
    with circuit.context as q:
        ops.X(q[0])

    return circuit


@pytest.fixture(scope="function")
def empty_circuit():
    """Empty circuit with no qubits, bits or operations."""
    circuit = Circuit()
    return circuit


@pytest.fixture(scope="function")
def two_qubit_op(monkeypatch):
    """Empty two-qubit operation."""

    class DummyOp(Operation):

        _num_qubits = 2

    monkeypatch.setattr(DummyOp, "__abstractmethods__", set())

    return DummyOp


@pytest.fixture(scope="function")
def two_qubit_parametric_op(monkeypatch):
    """Empty two-qubit parametric operation."""

    class DummyOp(ParametricOperation):

        _num_qubits = 2
        _num_params = 1

    monkeypatch.setattr(DummyOp, "__abstractmethods__", set())

    return DummyOp


@pytest.fixture(scope="function")
def two_qubit_controlled_op(monkeypatch):
    """Empty two-qubit controlled operation."""

    class DummyOp(ControlledOperation):

        _num_control = 1
        _num_target = 1
        _target_operation = ops.X

    monkeypatch.setattr(DummyOp, "__abstractmethods__", set())

    return DummyOp
