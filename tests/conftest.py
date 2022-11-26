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

from typing import Type

import pytest

import dwave.gate.operations as ops
from dwave.gate.circuit import Circuit, ParametricCircuit
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
def two_qubit_parametric_circuit():
    """ParametricCircuit with two qubits and two operation."""
    circuit = ParametricCircuit(2)

    with circuit.context as (p, q):
        ops.RX(p[0], q[0])
        ops.RY(p[1], q[1])

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
def empty_parametric_circuit():
    """Empty parametric circuit with no qubits, bits or operations."""
    circuit = ParametricCircuit()
    return circuit


@pytest.fixture(scope="function")
def two_qubit_op(monkeypatch):
    """Empty two-qubit operation."""

    class DummyOp(Operation):

        _num_qubits: int = 2

    monkeypatch.setattr(DummyOp, "__abstractmethods__", set())

    return DummyOp


@pytest.fixture(scope="function")
def two_qubit_parametric_op(monkeypatch):
    """Empty two-qubit parametric operation."""

    class DummyOp(ParametricOperation):

        _num_qubits: int = 2
        _num_params: int = 1

    monkeypatch.setattr(DummyOp, "__abstractmethods__", set())

    return DummyOp


@pytest.fixture(scope="function")
def two_qubit_controlled_op(monkeypatch):
    """Empty two-qubit controlled operation."""

    class DummyOp(ControlledOperation):

        _num_control: int = 1
        _num_target: int = 1
        _target_operation: Type[Operation] = ops.X

    monkeypatch.setattr(DummyOp, "__abstractmethods__", set())

    return DummyOp
