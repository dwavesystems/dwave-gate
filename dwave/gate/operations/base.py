# Confidential & Proprietary Information: D-Wave Systems Inc.
from __future__ import annotations

__all__ = [
    "Operation",
    "ParametricOperation",
    "ControlledOperation",
    "Measurement",
    "Barrier",
    "create_operation",
]

import copy
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from dwave.gate.circuit import Circuit, CircuitContext, ParametricCircuit
from dwave.gate.mixedproperty import mixedproperty
from dwave.gate.primitives import Qubit
from dwave.gate.registers import Variable
from dwave.gate.tools.unitary import build_controlled_unitary, build_unitary

if TYPE_CHECKING:
    from numpy.typing import NDArray


CustomOperation = TypeVar("CustomOperation", bound="Operation")
CustomParametricOperation = TypeVar("CustomParametricOperation", bound="ParametricOperation")


@overload
def create_operation(circuit: ParametricCircuit, name: Optional[str] = None) -> type[CustomParametricOperation]:  # type: ignore
    ...


@overload
def create_operation(circuit: Circuit, name: Optional[str] = None) -> type[CustomOperation]:  # type: ignore
    ...


def create_operation(circuit, name: Optional[str] = None) -> type[CustomOperation]:  # type: ignore
    """Create an operation from a circuit object.

    Takes the circuit operations and creates a new custom operation class inheriting either directly
    or indirectly from ``Operation``. The custrom operation will automatically implement the matrix
    property which constructs and stores the matrix represenation.

    Args:
        circuit: Circuit object out of which to create an operation.
        name: Name for the new operation. Usually the class name of the operation. Defaults to
            ``"CustomOperation"`` if no other name is given.

    Returns:
        Type: Class inheriting from the ``Operation`` class.
    """
    if circuit.parametric:
        superclass = ParametricOperation
    else:
        superclass = Operation

    class CustomOperation(superclass):  # type: ignore
        _num_qubits = circuit.num_qubits
        _num_params = circuit.num_parameters

        def __init__(self, *args, **kwargs) -> None:
            self._matrix = None
            super(CustomOperation, self).__init__(*args, **kwargs)

        def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
            """Converts the custom operation into an OpenQASM string.

            Note, the custom operation must be defined by the user before being used,
            using the lowecase representation of the name as the custom gate name.

            Returns:
                str: OpenQASM string representation of the customn operation.
            """
            if isinstance(name, str):
                new_name = name.lower()
            else:
                new_name = self.__class__.__name__.lower()

            if self.qubits:
                qubits = ", ".join(self._map_qubits(mapping))
                return f"{new_name} {qubits}"
            return new_name

        @mixedproperty
        def matrix(cls, self) -> NDArray:
            """The matrix representation of the custom operator.

            Note that this property call constructs, and caches, the matrix lazily
            by building the unitary based on the operations in the ``circuit``
            methods.
            """
            circuit_copy = copy.deepcopy(circuit)
            if circuit.parametric:
                parameters = self.parameters.copy()

                for i, op in enumerate(circuit_copy.circuit):
                    if isinstance(op, ParametricOperation):
                        for j, param in enumerate(op.parameters):
                            if isinstance(param, Variable):
                                circuit_copy.circuit[i].parameters[j] = parameters.pop(0)  # type: ignore
                            self._matrix

            return build_unitary(circuit_copy)

    if name:
        CustomOperation.__name__ = name

    # if circuit is parametric then ``self_required`` should be set to true in the mixedproperty
    # decorator; can be done by accessing the property and updating the attribute
    if circuit.parametric:
        CustomOperation.__dict__["matrix"]._self_required = True

    return CustomOperation


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

    def __init__(self, qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None) -> None:
        active_context = CircuitContext.active_context

        if qubits is not None:
            qubits = self._check_qubits(qubits)

        elif active_context is not None:
            raise TypeError("Qubits required when applying gate within context.")

        # must be set before appending operation to circuit
        self._qubits = qubits

        if active_context is not None and active_context.frozen is False:
            active_context.circuit.append(self)

    @classmethod
    def _check_qubits(cls, qubits: Union[Qubit, Sequence[Qubit]]) -> Tuple[Qubit, ...]:
        """Asserts size and type of the qubit(s) and returns the correct type.

        Args:
            qubits: Qubits to check.

        Returns:
            tuple: Sequence of qubits as a tuple.
        """
        if isinstance(qubits, Qubit):
            qubits = [qubits]

        if len(qubits) != cls.num_qubits:
            raise ValueError(
                f"Operation '{cls.label}' requires " f"{cls.num_qubits} qubits, got {len(qubits)}."
            )

        # cast to tuple for convention
        return tuple(qubits)

    def _map_qubits(
        self, mapping: Optional[Mapping[Hashable, Tuple[str, int]]] = None
    ) -> Sequence[str]:
        """Returns an OpenQASM 2.0 string representation of the operations qubits.

        Args:
            mapping: A mapping between the operations qubits and the qubits in the circuit. Must
                have qubits as keys and tuples, with the containing quantum register label and the
                qubits index within that register, as values.

        Returns:
            list: The OpenQASM 2.0 string representations of the operations qubits.
        """
        if mapping and self.qubits:
            return [f"{mapping[qb][0]}[{mapping[qb][1]}]" for qb in self.qubits]
        return [f"q[{i}]" for i in range(self.num_qubits)]

    def __call__(self, qubits: Optional[Sequence[Qubit]] = None) -> None:
        """Apply (or reapply) the operation within a context.

        Args:
            qubits: Qubits on which the operation should be applied. Only
                required if not already declared in operation.
        """
        qubits = qubits or self.qubits
        self.__class__(qubits)

    def __eq__(self, op: Operation) -> bool:
        """Returns whether two operations are considered equal."""
        name_eq = op.__class__.__name__ == self.__class__.__name__
        return name_eq and op.qubits == self.qubits

    def __str__(self) -> str:
        """Returns the operation representation."""
        return repr(self)

    def __repr__(self) -> str:
        """Returns the representation of the Operation object."""
        return f"<{self.__class__.__base__.__name__}: {self.label}, qubits={self.qubits}>"

    @mixedproperty
    def label(cls, self) -> str:
        """Qubit operation label."""
        if self and hasattr(self, "parameters"):
            params = f"({self.parameters})"
            return cls.__name__ + params  # type: ignore
        return cls.__name__  # type: ignore

    @mixedproperty
    def decomposition(cls) -> List[str]:
        """Decomposition of operation as list of operation labels."""
        decomposition = getattr(cls, "_decomposition", None)
        if not decomposition:
            raise NotImplementedError(
                "Decomposition not implemented for the " f"'{cls.label}' operation."
            )
        return decomposition

    @mixedproperty
    def num_qubits(cls) -> int:
        """Number of qubits that the operation supports."""
        if isinstance(cls._num_qubits, int):
            return cls._num_qubits

        raise AttributeError(f"Operations {cls.label} missing class attributes '_num_qubits'.")

    @abstractproperty
    def _num_qubits(cls) -> int:  # type: ignore
        """Abstract mixedproperty asserting the existence of a ``_num_qubits`` attribute."""

    @property
    def qubits(self) -> Optional[Tuple[Qubit, ...]]:
        """Qubits that the operation is applied to."""
        return self._qubits

    @qubits.setter
    def qubits(self, qubits: Sequence[Qubit]) -> None:
        """Set the qubits that the operation should be applied to."""
        if self.qubits is not None:
            warnings.warn(
                f"Changing qubits on which '{self}' is applied from {self._qubits} to {qubits}"
            )
        self._qubits = self._check_qubits(qubits)

    @abstractmethod
    def to_qasm(self, mapping: Optional[Mapping] = None) -> str:
        """Converts the operation into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the operation.
        """

    @abstractproperty
    def matrix(cls) -> NDArray:  # type: ignore
        """Returns the matrix representation of the operation.

        Returns:
            NDArray: Matrix representation of the operation.
        """


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
        qubits: Optional[Union[Qubit, Sequence[Qubit]]] = None,
    ):
        self._parameters = self._check_parameters(parameters)
        super(ParametricOperation, self).__init__(qubits)

    def __eq__(self, op: ParametricOperation) -> bool:
        """Returns whether two operations are considered equal."""
        param_eq = op.parameters == self.parameters
        return param_eq and super(ParametricOperation, self).__eq__(op)

    @property
    def parameters(self):
        """Parameters of the operation."""
        return self._parameters

    @mixedproperty
    def num_parameters(cls) -> int:
        """Number of parameters that the operation requires."""
        if isinstance(cls._num_params, int):
            return cls._num_params

        raise AttributeError(f"Operations {cls.label} missing class attribute '_num_params'.")

    @abstractproperty
    def _num_params(cls) -> int:  # type: ignore
        """Abstract mixedproperty asserting the existence of a ``_num_qubits`` attribute."""

    @classmethod
    def _check_parameters(cls, params: Sequence[complex]) -> List[complex]:
        """Asserts the size and type of the parameter(s) and returns the
        correct type.

        Args:
            params: Numeric parameters to check.

        Returns:
            list: Sequence of parameters as a list.
        """
        if not isinstance(params, Sequence):
            params = [params]

        if len(params) != cls._num_params:
            raise ValueError(
                f"Operation '{cls.label} requires "
                f"{cls._num_params} parameters, got {len(params)}."
            )

        # cast to list for convention
        return list(params)

    def __call__(self, qubits: Optional[Sequence[Qubit]] = None):
        """Apply (or reapply) the operation within a context.

        Args:
            qubits: Qubits on which the operation should be applied. Only
                required if not already declared in operation.
        """
        qubits = qubits or self.qubits
        self.__class__(self.parameters, qubits)


class ControlledOperation(Operation):
    """Generic controlled operation.

    Args:
        control: Qubit(s) on which the target operation is controlled.
        target: Qubit(s) on which the target operation is applied.
    """

    def __init__(
        self,
        control: Optional[Union[Qubit, Sequence[Qubit]]] = None,
        target: Optional[Union[Qubit, Sequence[Qubit]]] = None,
    ) -> None:

        if control:
            self._control = tuple([control] if isinstance(control, Qubit) else control)
        else:
            self._control = None

        if target:
            self._target = tuple([target] if isinstance(target, Qubit) else target)
        else:
            self._target = None

        # all qubits are stored in the 'qubits' attribute
        if self.control and self.target:
            qubits = tuple(chain.from_iterable((self.control, self.target)))
        else:
            qubits = None

        super(ControlledOperation, self).__init__(qubits)

    def __call__(
        self,
        control: Optional[Sequence[Qubit]] = None,
        target: Optional[Sequence[Qubit]] = None,
    ):
        """Apply (or reapply) the operation within a context.

        Args:
            control: Cpntrol qubits. Only required if not already declared in operation.
            target: Target qubits on which the operation should be applied. Only
                required if not already declared in operation.
        """
        control = control or self.control
        target = target or self.target

        self.__class__(control, target)  # type: ignore

    @abstractproperty
    def _num_control(cls):
        """Abstract mixedproperty asserting the existence of a ``_num_control`` attribute."""

    @abstractproperty
    def _num_target(cls):
        """Abstract mixedproperty asserting the existence of a ``_num_target`` attribute."""

    @abstractproperty
    def _target_operation(cls):
        """Abstract mixedproperty asserting the existence of a ``_target_operation`` attribute."""

    @mixedproperty
    def _num_qubits(cls) -> int:
        """Realizing the ``_num_qubits`` mixedproperty so that only ``_num_control`` and
        ``num_target`` attributes are required by subclasses."""
        # BUG: Unless an instance check for ints is here, 'mixedproperty' will complain
        # with 'TypeError: unsupported operand type(s) for +: 'function' and 'function''
        if isinstance(cls.num_control, int) and isinstance(cls.num_target, int):
            return cls.num_control + cls.num_target

        raise AttributeError(
            f"Operations {cls.label} missing class attributes '_num_control' and/or '_num_target'."
        )

    @mixedproperty
    def matrix(cls, self) -> NDArray:
        """The matrix representation of the controlled operation."""
        target_unitary = cls.target_operation.matrix
        num_qubits = getattr(cls, "num_qubits", 2)
        num_control = getattr(cls, "num_control", 1)

        if CircuitContext.active_context and self.control and self.target:
            control_idx = [
                CircuitContext.active_context.circuit.qubits.index(c) for c in self.control
            ]
            target_idx = [
                CircuitContext.active_context.circuit.qubits.index(t) for t in self.target
            ]
        else:
            control_idx = list(range(num_control))
            target_idx = list(range(num_control, num_qubits))

        matrix = build_controlled_unitary(
            control_idx, target_idx, target_unitary, num_qubits=num_qubits
        )
        return matrix

    @property
    def control(self) -> Optional[Tuple[Qubit]]:
        """Control qubit(s)."""
        return self._control

    @property
    def target(self) -> Optional[Tuple[Qubit]]:
        """Target qubit(s)."""
        return self._target

    @mixedproperty
    def num_control(cls) -> int:
        """Number of control qubit(s)."""
        assert cls._num_control
        return cls._num_control

    @mixedproperty
    def num_target(cls) -> int:
        """Number of target qubit(s)."""
        assert cls._num_target
        return cls._num_target

    @mixedproperty
    def target_operation(cls):
        """Target operation"""
        return cls._target_operation


class Measurement(Operation):
    """Class representing a measurement.

    Args:
        qubits: The qubits which should be measured. Only required when applying
            an measurement within a circuit context.
    """


class Barrier(Operation):
    """Class representing a barrier operation.

    Args:
        qubits: Qubits on which the barrier operation should be applied.
            Only required when applying a barrier operation within a circuit
            context.
    """
