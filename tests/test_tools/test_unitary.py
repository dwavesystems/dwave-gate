# Confidential & Proprietary Information: D-Wave Systems Inc.
import math

import numpy as np
import pytest

import dwave.gate.operations as ops
from dwave.gate import Circuit
from dwave.gate.tools import build_controlled_unitary, build_unitary


class TestBuildUnitary:
    """Test for the ``build_unitary`` function."""

    @pytest.mark.parametrize("num_qubits", [0, 1, 2, 3, 4])
    def test_empty_circuit(self, num_qubits):
        """Test building a unitary from an empty circuit."""
        circuit = Circuit(num_qubits)
        unitary = build_unitary(circuit)

        assert np.allclose(unitary, np.eye(2**num_qubits))

    @pytest.mark.parametrize(
        "operations, expected",
        [
            ([ops.X("q0")], np.array([[0, 1], [1, 0]])),
            ([ops.Y("q0")], np.array([[0, -1j], [1j, 0]])),
            ([ops.Z("q0")], np.array([[1, 0], [0, -1]])),
            ([ops.X("q0"), ops.Y("q0")], np.array([[-1j, 0], [0, 1j]])),
            (
                [ops.Hadamard("q0"), ops.X("q0")],
                np.array([[math.sqrt(2), -math.sqrt(2)], [math.sqrt(2), math.sqrt(2)]]) / 2,
            ),
            (
                [ops.X("q0"), ops.Hadamard("q0")],
                np.array([[math.sqrt(2), math.sqrt(2)], [-math.sqrt(2), math.sqrt(2)]]) / 2,
            ),
        ],
    )
    def test_single_qubit_circuit(self, operations, expected):
        """Test building a unitary from a single-qubit circuit"""
        circuit = Circuit(1)
        circuit.append(operations)

        assert np.allclose(build_unitary(circuit), expected)

    @pytest.mark.parametrize(
        "operations, expected",
        [
            (
                [ops.X("q0")],
                np.array(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                ),
            ),
            (
                [ops.Y("q1"), ops.X("q0")],
                np.array(
                    [
                        [0.0, 0.0, 0.0, -1.0j],
                        [0.0, 0.0, 1.0j, 0.0],
                        [0.0, -1.0j, 0.0, 0.0],
                        [1.0j, 0.0, 0.0, 0.0],
                    ]
                ),
            ),
            (
                [ops.Hadamard("q0"), ops.CZ("q0", "q1"), ops.Hadamard("q0")],
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                ),
            ),
        ],
    )
    def test_multi_qubit_circuit(self, operations, expected):
        """Test building a unitary from a multi-qubit circuit"""
        circuit = Circuit(2)
        circuit.append(operations)

        assert np.allclose(build_unitary(circuit), expected)


class TestBuildControlledUnitary:
    """Test for the ``build_controlled_unitary`` function."""

    @pytest.mark.parametrize(
        "control, target, unitary, expected",
        [
            (
                0,
                1,
                np.array([[0, 1], [1, 0]]),
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
            ),
            (
                1,
                0,
                np.array([[0, 1], [1, 0]]),
                np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
            ),
            (
                0,
                1,
                np.array([[-1j, 0], [0, 1]]),
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1]]),
            ),
            (
                0,
                2,
                np.array([[1, 0], [0, -1]]),
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, -1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, -1],
                    ]
                ),
            ),
            (
                2,
                0,
                np.array([[0, 1], [1, 0]]),
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                    ]
                ),
            ),
        ],
    )
    def test_single_control_single_target(self, control, target, unitary, expected):
        """Test building a controlled unitary with a single control and a single target."""
        result = build_controlled_unitary(control, target, unitary)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "control, target, unitary, expected",
        [
            (
                [0, 1],
                2,
                np.array([[0, 1], [1, 0]]),
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                    ]
                ),
            ),
            (
                [1, 2],
                0,
                np.array([[0, 1], [1, 0]]),
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                [0, 1],
                2,
                np.array([[-1j, 0], [0, 1]]),
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, -1j, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                ),
            ),
            (
                [0, 3],
                2,
                np.array([[1, 0], [0, -1]]),
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    ]
                ),
            ),
            (
                [3, 1],
                0,
                np.array([[0, 1], [1, 0]]),
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                [3, 2],
                0,
                np.array([[0, 1], [1, 0]]),
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
        ],
    )
    def test_multi_control_single_target(self, control, target, unitary, expected):
        """Test building a controlled unitary with a multi control and a single target."""
        result = build_controlled_unitary(control, target, unitary)
        assert np.allclose(result, expected)

    @pytest.mark.xfail(reason="Multiple targets not yet supported.")
    def test_single_control_multi_target(self):
        """Test building a controlled unitary with a single control and a multi target."""
        unitary = np.array([[0, 1], [1, 0]])
        build_controlled_unitary(0, [1, 2], unitary)

    @pytest.mark.xfail(reason="Multiple targets not yet supported.")
    def test_multi_control_multi_target(self):
        """Test building a controlled unitary with a multi control and a multi target."""
        unitary = np.array([[0, 1], [1, 0]])
        build_controlled_unitary([0, 1], [2, 3], unitary)

    @pytest.mark.parametrize(
        "control, target, num_qubits",
        [
            (0, 1, 1),
            (0, [3, 4], 2),
            (1, 0, 0),
            ([2, 3], 1, 2),
        ],
    )
    def test_incorrect_number_of_qubits(self, control, target, num_qubits):
        """Test that the correct exception is raised when controlling on/targeting a qubit outside the qubit range."""
        unitary = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="must be larger or equal to the largest qubit index"):
            build_controlled_unitary(control, target, unitary, num_qubits=num_qubits)

    @pytest.mark.parametrize(
        "control, target",
        [
            (0, 0),
            (1, 1),
            ([0, 1], 1),
            ([0, 3], [0, 2]),
        ],
    )
    def test_same_control_as_target(self, control, target):
        """Test that the correct exception is raised when using the same qubit as both control and target."""
        unitary = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="Control qubits and target qubit cannot be the same."):
            build_controlled_unitary(control, target, unitary)
