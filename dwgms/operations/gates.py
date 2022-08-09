import cmath
import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from dwgms.operations.operations import Operation
from dwgms.utils import classproperty

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

    @classproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Identity operator."""
        matrix = np.eye(2, dtype=np.float64)
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

    @classproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Pauli X operator."""
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
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

    @classproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Pauli Y operator."""
        matrix = np.array([[0.0, -1.0j], [1.0j, 0.0]])
        return matrix


class Z(Operation):
    """Pauli Z operator.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 1

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(Z, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the Pauli Z operator into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Pauli Z operator.
        """
        return f"z q[{self.qubits[0]}]"

    @classproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Pauli Z operator."""
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]])
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

    @classproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Hadamard operator."""
        matrix = math.sqrt(2) / 2 * np.array([[1.0, 1.0], [1.0, -1.0]])
        return matrix


#################################
# Parametric single-qubit gates #
#################################


class ParametricOperation(Operation):
    """Class for creating parametric operations.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
        parameters: Parameters for the operation. Required when constructing the
            matrix representation of the operation.
    """

    def __init__(
        self,
        parameters: Sequence[complex],
        qubits: Optional[Union[str, Sequence[str]]] = None,
    ):
        self._parameters = self._check_params(parameters)
        super().__init__(qubits)

    @property
    def parameters(self):
        """Parameters of the operation."""
        return self._parameters

    def _check_params(self, params):
        """Asserts the size and type of the parameter(s) and returns the
        correct type.

        Args:
            params: Parameters to check.

        Returns:
            list: Sequence of parameters as a list.
        """
        if isinstance(params, str) or not isinstance(params, Sequence):
            params = [params]

        if len(params) != self._num_params:
            raise ValueError(
                f"Operation '{self.__class__.__name__} requires "
                f"{self._num_params} parameters, got {len(params)}."
            )

        # cast to list for convention
        return list(params)


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

    def __init__(
        self, theta: float, qubits: Optional[Union[str, Sequence[str]]] = None
    ):
        super(RX, self).__init__([theta], qubits)

    def to_qasm(self) -> str:
        """Converts the Rotation-X operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Rotation-X operation.
        """
        theta = self.parameters[0]
        return f"rx({theta}) q[{self.qubits[0]}]"

    @classproperty
    def matrix(cls, self) -> NDArray:
        """The matrix representation of the Rotation-X operator."""
        # get parameters if called as instance method
        if self is None:
            raise ValueError("Require parameter values to construct matrix.")
        theta = self.parameters[0]

        diag_0 = math.cos(theta / 2)
        diag_1 = -1j * math.sin(theta / 2)
        matrix = np.array([[diag_0, diag_1], [diag_1, diag_0]])
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

    def __init__(
        self, theta: float, qubits: Optional[Union[str, Sequence[str]]] = None
    ):
        super(RY, self).__init__([theta], qubits)

    def to_qasm(self) -> str:
        """Converts the Rotation-Y operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Rotation-Y operation.
        """
        theta = self.parameters[0]
        return f"ry({theta}) q[{self.qubits[0]}]"

    @classproperty
    def matrix(cls, self) -> NDArray:
        """The matrix representation of the Rotation-Y operator."""
        # get parameters if called as instance method
        if self is None:
            raise ValueError("Require parameter values to construct matrix.")
        theta = self.parameters[0]

        diag_0 = math.cos(theta / 2)
        diag_1 = math.sin(theta / 2)
        matrix = np.array([[diag_0, -diag_1], [diag_1, diag_0]])
        return matrix


class RZ(ParametricOperation):
    """Rotation-Z operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
        parameters: Parameters for the operation. Required when constructing the
            matrix representation of the operation.
    """

    _num_qubits = 1
    _num_params = 1

    def __init__(
        self, theta: float, qubits: Optional[Union[str, Sequence[str]]] = None
    ):
        super(RZ, self).__init__([theta], qubits)

    def to_qasm(self) -> str:
        """Converts the Rotation-Z operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Rotation-Z operation.
        """
        theta = self.parameters[0]
        return f"rz({theta}) q[{self.qubits[0]}]"

    @classproperty
    def matrix(cls, self) -> NDArray:
        """The matrix representation of the Rotation-Z operator."""
        # get parameters if called as instance method
        if self is None:
            raise ValueError("Require parameter values to construct matrix.")
        theta = self.parameters[0]

        term_0 = cmath.exp(-1j * theta / 2)
        term_1 = cmath.exp(1j * theta / 2)
        matrix = np.array([[term_0, 0.0], [0.0, term_1]])
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
    _decomposition = [RZ, RY, RZ]

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

    @classproperty
    def matrix(cls, self) -> NDArray:
        """The matrix representation of the Rotation operator."""
        # get parameters if called as instance method
        if self is None:
            raise ValueError("Require parameter values to construct matrix.")
        theta_0 = self.parameters[0]
        theta_1 = self.parameters[1]
        theta_2 = self.parameters[2]

        mcos = math.cos(theta_1 / 2)
        msin = math.sin(theta_1 / 2)

        term_0 = cmath.exp(-1j * (theta_0 + theta_2) / 2) * mcos
        term_1 = -cmath.exp(-1j * (theta_0 - theta_2) / 2) * msin
        term_2 = cmath.exp(1j * (theta_0 - theta_2) / 2) * msin
        term_3 = cmath.exp(1j * (theta_0 + theta_2) / 2) * mcos

        matrix = np.array([[term_0, term_1], [term_2, term_3]])
        return matrix


#######################################
# Non-parametric multiple-qubit gates #
#######################################


class CX(Operation):
    """Controlled-X (CNOT) operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 2

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None) -> None:
        super(CX, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the Controlled X operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Controlled X operation.
        """
        return f"cx q[{self.qubits[0]}],  q[{self.qubits[1]}]"

    @classproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Controlled-X operation."""
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        return matrix


CNOT = CX
"""Controlled-NOT operation (alias for CX operation)"""


class CZ(Operation):
    """Controlled-Z operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    _num_qubits = 2

    def __init__(self, qubits: Optional[Union[str, Sequence[str]]] = None):
        super(CZ, self).__init__(qubits)

    def to_qasm(self) -> str:
        """Converts the Controlled-Z operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the Controlled-Z operation.
        """
        return f"cz q[{self.qubits[0]}],  q[{self.qubits[1]}]"

    @classproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the Controlled-Z operation."""
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ]
        )
        return matrix


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

    @classproperty
    def matrix(cls) -> NDArray:
        """The matrix representation of the SWAP operation."""
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return matrix


###################################
# Parametric multiple-qubit gates #
###################################
