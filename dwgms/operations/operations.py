# Confidential & Proprietary Information: D-Wave Systems Inc.
from __future__ import annotations

import cmath
import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from dwgms.mixedproperty import mixedproperty
from dwgms.operations.base import ControlledOperation, Operation, ParametricOperation

#####################################
# Non-parametric single-qubit gates #
#####################################


class Identity(Operation):
    """Identity operator.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 1

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(Identity, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the identity operator into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the identity operator.
        """
        return f"id q[{self.qubits[0]}]"

    @mixedproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Identity operator."""
        matrix = np.eye(2, dtype=complex)
        return matrix


class X(Operation):
    """Pauli X operator.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 1

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(X, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the Pauli X operator into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Pauli X operator.
        """
        return f"x q[{self.qubits[0]}]"

    @mixedproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Pauli X operator."""
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        return matrix


class Y(Operation):
    """Pauli Y operator.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 1

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(Y, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the Pauli Y operator into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Pauli Y operator.
        """
        return f"y q[{self.qubits[0]}]"

    @mixedproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Pauli Y operator."""
        matrix = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        return matrix


class Z(Operation):
    """Pauli Z operator.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 1
    _decomposition = ["Hadamard", "X", "Hadamard"]

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(Z, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the Pauli Z operator into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Pauli Z operator.
        """
        return f"z q[{self.qubits[0]}]"

    @mixedproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Pauli Z operator."""
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        return matrix


class Hadamard(Operation):
    """Hadamard operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 1

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(Hadamard, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the Hadamard operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Hadamard operation.
        """
        return f"h q[{self.qubits[0]}]"

    @mixedproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Hadamard operator."""
        matrix = math.sqrt(2) / 2 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
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

    _num_qubits = 1
    _num_params = 1

    def __init__(self, theta: float, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(RX, self).__init__(theta, qubits)

    def to_qasm(self) -> str:
        """Converts the Rotation-X operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Rotation-X operation.
        """
        theta = self.parameters[0]
        return f"rx({theta}) q[{self.qubits[0]}]"

    @mixedproperty(self_required=True)
    def matrix(cls, self) -> NDArray:
        """The matrix representation of the Rotation-X operator."""
        theta = self.parameters[0]

        diag_0 = math.cos(theta / 2)
        diag_1 = -1j * math.sin(theta / 2)
        matrix = np.array([[diag_0, diag_1], [diag_1, diag_0]], dtype=complex)
        return matrix


class RY(ParametricOperation):
    """Rotation-Y operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
        parameters: Parameters for the operation. Required when constructing the
            matrix representation of the operation.
    """

    _num_qubits = 1
    _num_params = 1

    def __init__(self, theta: float, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(RY, self).__init__(theta, qubits)

    def to_qasm(self) -> str:
        """Converts the Rotation-Y operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Rotation-Y operation.
        """
        theta = self.parameters[0]
        return f"ry({theta}) q[{self.qubits[0]}]"

    @mixedproperty(self_required=True)
    def matrix(cls, self) -> NDArray:
        """The matrix representation of the Rotation-Y operator."""
        theta = self.parameters[0]

        diag_0 = math.cos(theta / 2)
        diag_1 = math.sin(theta / 2)
        matrix = np.array([[diag_0, -diag_1], [diag_1, diag_0]], dtype=complex)
        return matrix


class RZ(ParametricOperation):
    """Rotation-Z operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
        theta: Parameter for the operation. Required when constructing the
            matrix representation of the operation.
    """

    _num_qubits = 1
    _num_params = 1

    def __init__(self, theta: float, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(RZ, self).__init__(theta, qubits)

    def to_qasm(self) -> str:
        """Converts the Rotation-Z operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Rotation-Z operation.
        """
        theta = self.parameters[0]
        return f"rz({theta}) q[{self.qubits[0]}]"

    @mixedproperty(self_required=True)
    def matrix(cls, self) -> NDArray:
        """The matrix representation of the Rotation-Z operator."""
        theta = self.parameters[0]

        term_0 = cmath.exp(-1j * theta / 2)
        term_1 = cmath.exp(1j * theta / 2)
        matrix = np.array([[term_0, 0.0], [0.0, term_1]], dtype=complex)
        return matrix


class Rotation(ParametricOperation):
    """Rotation operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
        parameters: Parameters for the operation. Required when constructing the
            matrix representation of the operation.
    """

    _num_qubits = 1
    _num_params = 3
    _decomposition = ["RZ", "RY", "RZ"]

    def __init__(
        self,
        parameters: Sequence[float],
        qubits: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        super(Rotation, self).__init__(parameters, qubits)

    def to_qasm(self) -> str:
        """Converts the Rotation operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Rotation operation.
        """
        return f"rz q[{self.qubits[0]}]\nry q[{self.qubits[0]}]\nrz q[{self.qubits[0]}]"

    @mixedproperty(self_required=True)
    def matrix(cls, self) -> NDArray:
        """The matrix representation of the Rotation operator."""
        beta, gamma, delta = self.parameters

        mcos = math.cos(gamma / 2)
        msin = math.sin(gamma / 2)

        term_0 = cmath.exp(-0.5j * (beta + delta)) * mcos
        term_1 = -cmath.exp(0.5j * (beta - delta)) * msin
        term_2 = cmath.exp(-0.5j * (beta - delta)) * msin
        term_3 = cmath.exp(0.5j * (beta + delta)) * mcos

        matrix = np.array([[term_0, term_1], [term_2, term_3]], dtype=complex)
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

    _num_control = 1
    _num_target = 1
    _target_operation = X

    def __init__(self, control: Optional[str] = None, target: Optional[str] = None) -> None:
        super(CX, self).__init__(op=self._target_operation, control=control, target=target)

    def to_qasm(self) -> str:
        """Converts the Controlled X operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Controlled X operation.
        """
        return f"cx q[{self.qubits[0]}],  q[{self.qubits[1]}]"


CNOT = CX
"""Controlled-NOT operation (alias for the CX operation)"""


class CZ(ControlledOperation):
    """Controlled-Z operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_control = 1
    _num_target = 1
    _target_operation = Z

    def __init__(self, control: Optional[str] = None, target: Optional[str] = None) -> None:
        super(CZ, self).__init__(op=self._target_operation, control=control, target=target)

    def to_qasm(self) -> str:
        """Converts the Controlled-Z operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Controlled-Z operation.
        """
        return f"cz q[{self.qubits[0]}],  q[{self.qubits[1]}]"


class SWAP(Operation):
    """SWAP operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 2

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(SWAP, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the SWAP operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the SWAP operation.
        """
        raise NotImplementedError("OpenQASM 2.0 does not support the SWAP gate.")

    @mixedproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the SWAP operation."""
        # TODO: add support for larger matrix states
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )
        return matrix


class CSWAP(Operation):
    """CSWAP operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 3

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(CSWAP, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the SWAP operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the SWAP operation.
        """
        raise NotImplementedError("OpenQASM 2.0 does not support the CSWAP gate.")

    @mixedproperty
    def matrix(cls) -> NDArray:
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
            dtype=complex,
        )
        return matrix


###################################
# Parametric multiple-qubit gates #
###################################
