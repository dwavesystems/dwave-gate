# Confidential & Proprietary Information: D-Wave Systems Inc.
from __future__ import annotations

__all__ = [
    "Identity",
    "X",
    "Y",
    "Z",
    "Hadamard",
    "RX",
    "RY",
    "RZ",
    "Rotation",
    "CX",
    "CNOT",
    "CZ",
    "SWAP",
    "CSWAP",
]

import cmath
import math
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from dwave.gate.mixedproperty import mixedproperty
from dwave.gate.operations.base import ControlledOperation, Operation, ParametricOperation
from dwave.gate.primitives import Qubit

#####################################
# Non-parametric single-qubit gates #
#####################################


class Identity(Operation):
    """Identity operator.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits: int = 1

    def __init__(self, qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None):
        super(Identity, self).__init__(qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the identity operator into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the identity operator.
        """
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"id {qubits[0]}"
        return "id"

    @mixedproperty
    def matrix(cls) -> NDArray[np.complex128]:
        """The matrix representation of the Identity operator."""
        matrix = np.eye(2, dtype=np.complex128)
        return matrix


class X(Operation):
    """Pauli X operator.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits: int = 1

    def __init__(self, qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None):
        super(X, self).__init__(qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Pauli X operator into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Pauli X operator.
        """
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"x {qubits[0]}"
        return "x"

    @mixedproperty
    def matrix(cls) -> NDArray[np.complex128]:
        """The matrix representation of the Pauli X operator."""
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        return matrix


class Y(Operation):
    """Pauli Y operator.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits: int = 1

    def __init__(self, qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None):
        super(Y, self).__init__(qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Pauli Y operator into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Pauli Y operator.
        """
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"y {qubits[0]}"
        return "y"

    @mixedproperty
    def matrix(cls) -> NDArray[np.complex128]:
        """The matrix representation of the Pauli Y operator."""
        matrix = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        return matrix


class Z(Operation):
    """Pauli Z operator.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits: int = 1
    _decomposition = ["Hadamard", "X", "Hadamard"]

    def __init__(self, qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None):
        super(Z, self).__init__(qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Pauli Z operator into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Pauli Z operator.
        """
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"z {qubits[0]}"
        return "z"

    @mixedproperty
    def matrix(cls) -> NDArray[np.complex128]:
        """The matrix representation of the Pauli Z operator."""
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        return matrix


class Hadamard(Operation):
    """Hadamard operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits: int = 1

    def __init__(self, qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None):
        super(Hadamard, self).__init__(qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Hadamard operation into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Hadamard operation.
        """
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"h {qubits[0]}"
        return "h"

    @mixedproperty
    def matrix(cls) -> NDArray[np.complex128]:
        """The matrix representation of the Hadamard operator."""
        matrix = math.sqrt(2) / 2 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
        return matrix


#################################
# Parametric single-qubit gates #
#################################


class RX(ParametricOperation):
    """Rotation-X operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
        parameters: Parameters for the operation. Required when constructing the
            matrix representation of the operation.
    """

    _num_qubits: int = 1
    _num_params: int = 1

    def __init__(
        self,
        theta: Union[float, Sequence[float]],
        qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None,
    ):
        super(RX, self).__init__([theta] if isinstance(theta, float) else theta, qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Rotation-X operation into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Rotation-X operation.
        """
        theta = self.parameters[0]
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"rx({theta}) {qubits[0]}"
        return f"rx({theta})"

    @mixedproperty(self_required=True)
    def matrix(cls, self) -> NDArray[np.complex128]:
        """The matrix representation of the Rotation-X operator."""
        theta = self.parameters[0]

        diag_0 = math.cos(theta / 2)
        diag_1 = -1j * math.sin(theta / 2)
        matrix = np.array([[diag_0, diag_1], [diag_1, diag_0]], dtype=np.complex128)
        return matrix


class RY(ParametricOperation):
    """Rotation-Y operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
        parameters: Parameters for the operation. Required when constructing the
            matrix representation of the operation.
    """

    _num_qubits: int = 1
    _num_params: int = 1

    def __init__(
        self,
        theta: Union[float, Sequence[float]],
        qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None,
    ):
        super(RY, self).__init__([theta] if isinstance(theta, float) else theta, qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Rotation-Y operation into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Rotation-Y operation.
        """
        theta = self.parameters[0]
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"ry({theta}) {qubits[0]}"
        return f"ry({theta})"

    @mixedproperty(self_required=True)
    def matrix(cls, self) -> NDArray[np.complex128]:
        """The matrix representation of the Rotation-Y operator."""
        theta = self.parameters[0]

        diag_0 = math.cos(theta / 2)
        diag_1 = math.sin(theta / 2)
        matrix = np.array([[diag_0, -diag_1], [diag_1, diag_0]], dtype=np.complex128)
        return matrix


class RZ(ParametricOperation):
    """Rotation-Z operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
        theta: Parameter for the operation. Required when constructing the
            matrix representation of the operation.
    """

    _num_qubits: int = 1
    _num_params: int = 1

    def __init__(
        self,
        theta: Union[float, Sequence[float]],
        qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None,
    ):
        super(RZ, self).__init__([theta] if isinstance(theta, float) else theta, qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Rotation-Z operation into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Rotation-Z operation.
        """
        theta = self.parameters[0]
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"rz({theta}) {qubits[0]}"
        return f"rz({theta})"

    @mixedproperty(self_required=True)
    def matrix(cls, self) -> NDArray[np.complex128]:
        """The matrix representation of the Rotation-Z operator."""
        theta = self.parameters[0]

        term_0 = cmath.exp(-1j * theta / 2)
        term_1 = cmath.exp(1j * theta / 2)
        matrix = np.array([[term_0, 0.0], [0.0, term_1]], dtype=np.complex128)
        return matrix


class Rotation(ParametricOperation):
    """Rotation operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
        parameters: Parameters for the operation. Required when constructing the
            matrix representation of the operation.
    """

    _num_qubits: int = 1
    _num_params: int = 3
    _decomposition = ["RZ", "RY", "RZ"]
    _qasm_decl: str = (
        "gate rot(beta, gamma, delta) { rz(beta) q[0]; ry(gamma) q[0]; rz(delta) q[0]; }"
    )

    def __init__(
        self,
        parameters: Sequence[float],
        qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None,
    ) -> None:
        super(Rotation, self).__init__(parameters, qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Rotation operation into an OpenQASM string.

        Note, the Rotation operation must be defined by decomposing into existing gates, e.g.,
        using RY and RZ gates as follows:

        .. code-block::

            gate rot(beta, gamma, delta) { rz(beta) q[0]; ry(gamma) q[0]; rz(delta) q[0]; }

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Rotation operation.
        """
        beta, gamma, delta = self.parameters
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"rot({beta}, {gamma}, {delta}) {qubits[0]}"
        return f"rot({beta}, {gamma}, {delta})"

    @mixedproperty(self_required=True)
    def matrix(cls, self) -> NDArray[np.complex128]:
        """The matrix representation of the Rotation operator."""
        beta, gamma, delta = self.parameters

        mcos = math.cos(gamma / 2)
        msin = math.sin(gamma / 2)

        term_0 = cmath.exp(-0.5j * (beta + delta)) * mcos
        term_1 = -cmath.exp(0.5j * (beta - delta)) * msin
        term_2 = cmath.exp(-0.5j * (beta - delta)) * msin
        term_3 = cmath.exp(0.5j * (beta + delta)) * mcos

        matrix = np.array([[term_0, term_1], [term_2, term_3]], dtype=np.complex128)
        return matrix


#######################################
# Non-parametric multiple-qubit gates #
#######################################


class CX(ControlledOperation):
    """Controlled-X (CNOT) operation.

    Args:
        control: Qubit on which the target operation ``X`` is controlled.
        target: Qubit on which the target operation ``X`` is applied.
    """

    _num_control: int = 1
    _num_target: int = 1
    _target_operation: type[Operation] = X

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Controlled X operation into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Controlled X operation.
        """
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"cx {qubits[0]}, {qubits[1]}"
        return "cx"


CNOT = CX
"""Controlled-NOT operation (alias for the CX operation)"""


class CZ(ControlledOperation):
    """Controlled-Z operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_control: int = 1
    _num_target: int = 1
    _target_operation: type[Operation] = Z

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the Controlled-Z operation into an OpenQASM string.

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the Controlled-Z operation.
        """
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"cz {qubits[0]}, {qubits[1]}"
        return "cz"


class SWAP(Operation):
    """SWAP operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits: int = 2
    _qasm_decl: str = "gate swap a, b { cx b, a; cx a, b; cx b, a; }"

    def __init__(self, qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None):
        super(SWAP, self).__init__(qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the SWAP operation into an OpenQASM string.

        Note, the SWAP operation must be defined by decomposing into existing gates, e.g., using
        CNOT gates as follows:

        .. code-block::

            gate swap a, b { cx b, a; cx a, b; cx b, a; }

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the SWAP operation.
        """
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"swap {qubits[0]}, {qubits[1]}"
        return "swap"

    @mixedproperty
    def matrix(cls) -> NDArray[np.complex128]:
        """The matrix representation of the SWAP operation."""
        # TODO: add support for larger matrix states
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.complex128,
        )
        return matrix


class CSWAP(Operation):
    """CSWAP operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits: int = 3
    _qasm_decl: str = "gate cswap c, a, b { cx b, a; ccx c, a, b; cx b, a; }"

    def __init__(self, qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None):
        super(CSWAP, self).__init__(qubits)

    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the CSWAP operation into an OpenQASM string.

        Note, the CSWAP operation must be defined by decomposing into existing gates, e.g., using
        CNOT and Toffoli gates as follows:

        .. code-block::

            gate cswap c, a, b { cx b, a; ccx c, a, b; cx b, a; }

        Args:
            mapping: Optional mapping between qubits and their indices in the circuit.

        Returns:
            str: OpenQASM string representation of the SWAP operation.
        """
        if self.qubits:
            qubits = self._map_qubits(mapping)
            return f"cswap {qubits[0]}, {qubits[1]}, {qubits[2]}"
        return "cswap"

    @mixedproperty
    def matrix(cls) -> NDArray[np.complex128]:
        """The matrix representation of the SWAP operation."""
        # TODO: add support for larger matrix states
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.complex128,
        )
        return matrix


###################################
# Parametric multiple-qubit gates #
###################################
