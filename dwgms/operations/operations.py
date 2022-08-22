# Confidential & Proprietary Information: D-Wave Systems Inc.
from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Hashable, Optional, Sequence, Union

from dwgms.circuit import CircuitContext
from dwgms.utils import abstractclassproperty, classproperty

if TYPE_CHECKING:
    from numpy.typing import NDArray

# IDEA: use a functional approach, instead of an object oriented approach, for
# utility functions such as 'broadcast', 'decompose' and 'get_matrix'.


class ABCLockedAttr(ABCMeta):
    """Metaclass to lock certain class attributes from being overwritten
    with instance attributes. Used by the ``Operation`` abstract class."""

    locked_attributes = [
        "matrix",
    ]

    def __setattr__(self, attr, value) -> None:
        if attr in self.locked_attributes:
            raise ValueError(f"Cannot set class attribute '{attr}' to '{value}'.")

        super(ABCLockedAttr, self).__setattr__(attr, value)


class Operation(metaclass=ABCLockedAttr):
    """Class representing a classical or quantum operation.

    Args:
        qubits: Qubits on which the operation should be applied. Only
            required when applying an operation within a circuit context.
    """

    def __init__(
        self, qubits: Optional[Union[Hashable, Sequence[Hashable]]] = None
    ) -> None:
        active_context = CircuitContext.active_context

        if qubits is not None:
            qubits = self._check_qubits(qubits)

        elif active_context is not None:
            raise TypeError(
                "Qubits required when applying gate withing context. Did you "
                "check that there are enough qubits in the quantum register?"
            )

        # must be set before appending operation to circuit
        self._qubits = qubits

        if active_context is not None:
            active_context.circuit.append(self)

    def _check_qubits(self, qubits: Sequence[Hashable]) -> Sequence[Hashable]:
        """Asserts size and type of the qubit(s) and returns the correct type.

        Args:
            qubits: Qubits to check.

        Returns:
            tuple: Sequence of qubits as a tuple.
        """
        # TODO: update to check for single qubit instead of str
        if isinstance(qubits, str) or not isinstance(qubits, Sequence):
            qubits = [qubits]

        if len(qubits) != self.num_qubits:
            raise ValueError(
                f"Operation '{self.__class__.__name__} requires "
                f"{self.num_qubits} qubits, got {len(qubits)}."
            )

        # cast to tuple for convention
        return tuple(qubits)

    @classmethod
    def broadcast(
        cls,
        qubits: Sequence[Hashable],
        params: Sequence[complex] = None,
        method: str = "layered",
    ) -> None:
        """Broadcasts an operation over one or more qubits.

        Args:
            qubits: Qubits to broadcast the operation over.
            params: Operation parameters, if required by operation.
            method: Which method to use for broadcasting (defaults to
                ``"layered"``). Currently supports the following methods:

                - layered: Applies the operation starting at the first qubit,
                  incrementing by a single qubit per round; e.g., a 2-qubit gate
                  applied to 3 qubits would apply it first to (0, 1), then (1, 2).
                - parallel: Applies the operation starting at the first qubit,
                  incrementing by the number of supported qubits for that gate;
                  e.g., a 2-qubit gate applied to 5 qubits would apply it first
                  to (0, 1), then (2, 3) and not apply anything to qubit 4.
        Raises:
            ValueError: If an unsupported method is requested.
        """
        # add parameters to operation call if required
        if params:
            append = lambda start, end: cls(params, qubits[start:end])
        else:
            append = lambda start, end: cls(qubits[start:end])

        if method == "layered":
            for i, _ in enumerate(qubits[: -cls._num_qubits + 1 or None]):
                append(i, i + cls._num_qubits)
        elif method == "parallel":
            for i, _ in enumerate(qubits[:: cls._num_qubits]):
                start, end = cls._num_qubits * i, cls._num_qubits * (i + 1)
                if end <= len(qubits):
                    append(start, end)
        else:
            raise ValueError(f"'{method}' style not supported.")

    def __str__(self) -> str:
        """Returns the operation representation."""
        return repr(self)

    def __repr__(self) -> str:
        """Returns the representation of the Operation object."""
        return (
            f"<{self.__class__.__base__.__name__}: {self.label}, qubits={self.qubits}>"
        )

    @classproperty
    def label(cls, self) -> str:
        """Qubit operation label."""
        if self and hasattr(self, "parameters"):
            params = f"({self.parameters})"
            return cls.__name__ + params
        return cls.__name__

    @classproperty
    def num_qubits(cls) -> int:
        """Number of qubits that the operation supports."""
        return cls._num_qubits

    @property
    def qubits(self) -> Sequence[Hashable]:
        """Qubits that the operation is applied to."""
        return self._qubits

    @qubits.setter
    def qubits(self, qubits: Sequence[Hashable]) -> None:
        """Set the qubits that the operation should be applied to."""
        if self.qubits is not None:
            warnings.warn(
                f"Changing qubits on which '{self}' is applied from {self._qubits} to {qubits}"
            )
        self._qubits = self._check_qubits(qubits)

    @abstractmethod
    def to_qasm(self):
        """Converts the operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the operation.
        """
        pass

    @abstractclassproperty
    def matrix(cls) -> NDArray:
        """Returns the matrix representation of the operation.

        Returns:
            NDArray: Matrix representation of the operation.
        """
        pass

    @classproperty
    def decomposition(cls, self) -> Sequence["Operation"]:
        """Returns the decomposition of operation."""
        if not getattr(cls, "_decomposition", None):
            raise NotImplementedError(
                "Decomposition not implemented for the " f"'{cls.__name__}' operation."
            )

        # if applying a decomposition, remove the applied (un-decomposed) gate first
        if CircuitContext.active_context is not None:
            del CircuitContext.active_context.circuit.circuit[-1]

        if self is not None:
            if self.parameters and not self.qubits:
                return [
                    op(self.parameters[i]) for i, op in enumerate(cls._decomposition)
                ]
            if self.parameters and self.qubits:
                return [
                    op(self.parameters[i], self.qubits[0])
                    for i, op in enumerate(cls._decomposition)
                ]
            if not self.parameters and self.qubits:
                return [op(self.qubits[0]) for op in cls._decomposition]
        return cls._decomposition


class Measure(Operation):
    """Class representing a measurement.

    Args:
        qubits: The qubits which should be measured. Only required when applying
            an measurement within a circuit context.
    """

    def __init__(self, qubits: Optional[Union[Hashable, Sequence[Hashable]]] = None):
        super(Measure, self).__init__(qubits)

    def to_qasm(self):
        """Converts the measurement into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the measurement.
        """
        return "measure"


class Barrier(Operation):
    """Class representing a barrier operation.

    Args:
        qubits: Qubits on which the barrier operation should be applied.
            Only required when applying a barrier operation within a circuit
            context.
    """

    def __init__(self, qubits: Optional[Union[Hashable, Sequence[Hashable]]] = None):
        super(Barrier, self).__init__(qubits)

    def to_qasm(self):
        """Converts the barrier into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the barrier.
        """
        return "barrier"
