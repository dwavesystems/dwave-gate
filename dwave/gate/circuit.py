# Copyright 2022-2023 D-Wave Systems Inc.
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

"""Circuit and context manager classes.

Contains classes to construct and handle quantum circuits. The quantum circuits contain the quantum
operations and instructions for running the circuits on simulators or hardware. See
:py:mod:`dwave.gate.operations` for details on which operations are supported.
"""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "CircuitError",
    "Circuit",
    "ParametricCircuit",
    "CircuitContext",
    "ParametricCircuitContext",
]

import copy
from functools import cached_property
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Callable,
    ContextManager,
    Dict,
    Hashable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from dwave.gate.mixedproperty import mixedproperty
from dwave.gate.primitives import Bit, Qubit
from dwave.gate.registers.registers import (
    ClassicalRegister,
    QuantumRegister,
    SelfIncrementingRegister,
)

if TYPE_CHECKING:
    from dwave.gate.operations.base import Operation

Qubits = Union[Qubit, Sequence[Qubit]]
Bits = Union[Bit, Sequence[Bit]]


class CircuitError(Exception):
    """Exception to be raised when there is an error with a Circuit."""


class Circuit:
    """Class to build and manipulate quantum circuits.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_bits: Number of classical bits in the circuit.
    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        num_bits: Optional[int] = None,
    ) -> None:
        self._circuit: List[Operation] = []
        self._circuit_context: Optional[CircuitContext] = None

        # registers for quantum and classical bits
        self._qregisters: Dict[Hashable, QuantumRegister] = dict()
        self._cregisters: Dict[Hashable, ClassicalRegister] = dict()

        if num_qubits is not None:
            self.add_qregister(num_qubits=num_qubits)

        if num_bits is not None:
            self.add_cregister(num_bits=num_bits)

        self._state: NDArray = None
        self._density_matrix: NDArray = None

        self._locked: bool = False

    def __call__(self, qubits: Qubits, bits: Optional[Bits] = None) -> None:
        """Apply all the operations in the circuit within a circuit context.

        Args:
            qubits: Qubits on which the circuit operations should be applied. The qubits used in
                the circuit will be exchanged with the corresponding ones (e.g., with the same
                index as) in the active context.
            bits: Bits in which to store measurement values if measurements have been made.

        Raises:
            ValueError: If an invalid number of qubits is passed.
            CircuitError: If called outside of an active context.
        """
        if CircuitContext.active_context is None:
            raise CircuitError("Can only apply circuit object inside a circuit context.")
        if CircuitContext.active_context.circuit is self:
            raise TypeError("Cannot apply circuit in its own context.")

        if isinstance(qubits, Qubit):
            qubits = [qubits]

        if len(qubits) != len(self.qubits):
            raise ValueError(f"Circuit requires {len(self.qubits)} qubits, got {len(qubits)}.")

        if bits:
            if isinstance(bits, Bit):
                bits = [bits]

            if len(bits) != len(self.bits):
                raise ValueError(f"Circuit requires {len(self.bits)} bits, got {len(bits)}.")

            bit_map = dict(zip(self.bits, bits))

        qubit_map = dict(zip(self.qubits, qubits))

        for op in self.circuit:
            mapped_qubits = [qubit_map[qb] for qb in op.qubits or []]

            # NOTE: avoid circular imports; needed to check operation type
            from dwave.gate.operations.base import (
                ControlledOperation,
                Measurement,
                ParametricOperation,
            )

            if isinstance(op, Measurement):
                if bits:
                    mapped_bits = [bit_map[qb] for qb in op.bits or []]
                    op.__class__(qubits=mapped_qubits) | mapped_bits
                    continue
                else:
                    warnings.warn(
                        "Measurements not stored in circuit bits. "
                        "Must pass bits to circuit call."
                    )

            if isinstance(op, ParametricOperation):
                op.__class__(op.parameters, qubits=mapped_qubits)
            elif isinstance(op, ControlledOperation):
                op.__class__(*mapped_qubits)
            else:
                op.__class__(qubits=mapped_qubits)

    @property
    def qregisters(self) -> Mapping[Hashable, QuantumRegister]:
        """Quantum registers of the circuit.

        Returns a dictionary with quantum register labels as keys and
        :class:`QuantumRegister` objects as values.
        """
        return self._qregisters

    @property
    def cregisters(self) -> Mapping[Hashable, ClassicalRegister]:
        """Classical registers of the circuit.

        Returns a dictionary with classical register labels as keys and
        :class:`ClassicalRegister` objects as values.
        """
        return self._cregisters

    @property
    def circuit(self) -> List[Operation]:
        """Circuit containing the applied operations."""
        return self._circuit

    def append(self, operation: Operation) -> None:
        """Appends an operation to the circuit.

        Args:
            operation: Operation to append to the circuit.
        """
        if self.is_locked():
            raise CircuitError(
                "Circuit is locked and no more operations can be appended. To "
                "unlock the circuit, call 'Circuit.unlock()' first."
            )
        if not self.parametric:
            # if a parametric operation is called within a non-parameteric circuit, all variables
            # should be replaced by their corresponding parameter values; eval does that
            eval = getattr(operation, "eval", None)
            operation = eval(inplace=True) if eval else operation

        for q in operation.qubits or []:
            if q not in self.qubits:
                raise ValueError(f"Qubit '{q}' not in circuit.")

        self._circuit.append(operation)

    def extend(self, operations: Sequence[Operation]) -> None:
        """Appends a sequence of operations to the circuit.

        Args:
            operations: Operations to append to the circuit.
        """
        for op in operations:
            self.append(op)

    def remove(self, op: Operation) -> None:
        """Removes the operation from the circuit.

        Args:
            op: Operation to remove.
        """
        try:
            idx = self.circuit.index(op)
        except ValueError as e:
            raise ValueError(f"Operation '{op}' not in circuit.") from e

        del self.circuit[idx]

    @cached_property
    def qubits(self) -> QuantumRegister:
        """Qubits handled by the circuit."""
        qubit_reg = QuantumRegister()
        for qreg in self.qregisters.values():
            qubit_reg += qreg

        return qubit_reg

    @cached_property
    def bits(self) -> ClassicalRegister:
        """Classical bits handled by the circuit."""
        bit_reg = ClassicalRegister()
        for creg in self.cregisters.values():
            bit_reg += creg

        return bit_reg

    @property
    def state(self) -> Optional[NDArray]:
        """The resulting state after simulating the circuit."""
        if self._state is None and self._density_matrix is not None:
            raise CircuitError("State is mixed. Use 'Circuit.density_matrix' to access.")
        return self._state

    @property
    def density_matrix(self) -> Optional[NDArray]:
        """The density matrix representation of the state."""
        if self._state is not None and self._density_matrix is None:
            self._density_matrix = self._state.reshape(-1, 1) @ self._state.reshape(1, -1)
        return self._density_matrix

    def set_state(self, state: NDArray, force: bool = False, normalize: bool = False) -> None:
        """Set the state of the circuit (used primarily with simulator).

        Sets either the ``Circuit.state`` attribute or the ``Circuit.density_matrix`` attribute
        depending on the shape of the state.

        Args:
            state: The state (pure or mixed) to set.
            force: Whether to overwrite an already set state or not.
            normalize: Whether to normalize the state before setting.
        """
        state_is_set = self._state is not None or self._density_matrix is not None
        if not force and state_is_set:
            raise CircuitError("State already set. Use 'force=True' to force set new state.")

        if normalize:
            state = state / np.linalg.norm(state)

        self._assert_state(state)

        if state.ndim == 1:
            self._state = state
            self._density_matrix = None

        else:  # if state.ndim == 2:
            self._state = None
            self._density_matrix = state

    def _assert_state(self, state: NDArray) -> None:
        """Check whether a state is the correct shape and is normalized."""
        size = 2 << (self.num_qubits - 1)
        if state.shape != (size,) and state.shape != (size, size):
            raise ValueError(f"State has incorrect shape. Should have size {size}.")

        if not np.isclose(np.linalg.norm(state), 1):
            raise ValueError("State is not normalized.")

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return len(self.qubits)

    @property
    def num_bits(self) -> int:
        """Number of bits in the circuit."""
        return len(self.bits)

    @property
    def parametric(self) -> bool:
        """Whether the circuit has parameter variables."""
        # base circuit is never parametric
        return False

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the circuit."""
        # base circuit has no parameters
        return 0

    @property
    def context(self) -> CircuitContext:
        """Circuit context used to apply operations to the circuit.

        A :class:`dwave.gate.circuit.Registers` is returned when entering the context, which
        contains the quantum and classical registers, named ``q`` and ``c`` respectively. These
        contain all qubits and classical bits respectively and can be used to reference operation
        applications and measurement values. All operations initialized within the context are
        automatically applied to the circuit unless called within a frozen context.
        """
        if self._circuit_context is None:
            self._circuit_context = CircuitContext(circuit=self)
        return self._circuit_context

    def lock(self) -> None:
        """Locks the circuit so that no more operations can be applied."""
        self._locked = True

    def unlock(self) -> None:
        """Unlocks the circuit allowing for further operations to be applied."""
        self._locked = False

    def is_locked(self) -> bool:
        """Whether the circuit is locked or not."""
        return self._locked

    def reset(self, keep_registers: bool = True) -> None:
        """Resets the circuit so that it can be reused.

        Args:
            keep_registers: If ``False``, deletes the quantum and classical
                registers, removing all the current qubits in the circuit,
                including those created at initialization (defaults to
                ``True``).
        """
        self._circuit.clear()
        self._circuit_context = None
        if not keep_registers:
            self._qregisters.clear()
            self._cregisters.clear()
        else:
            for bit in self.bits:
                bit.reset()

        self.unlock()

    def add_qubit(
        self, qubit: Optional[Qubit] = None, qreg_label: Optional[Hashable] = None
    ) -> None:
        """Add a single qubit to a quantum register in the circuit.

        Args:
            qubit: Qubit to add to the circuit. If ``None``, then a new qubit is created.
            qreg_label: Label for the quantum register to which the new qubit should be
                appended (defaults to 'r' followed by a random integer ID number).
        """
        if qubit is not None and not isinstance(qubit, Qubit):
            raise TypeError(f"Can only add qubits to circuit. Got '{type(qubit).__name__}'.")

        if qreg_label is not None:
            if qreg_label not in self.qregisters:
                self.add_qregister(label=qreg_label)
        else:
            if not self.qregisters:
                self.add_qregister()
            qreg_label = list(self.qregisters)[0]

        # NOTE: same qubit in different registers NOT allowed
        if qubit in self.qubits:
            raise ValueError(f"Qubit '{qubit}' already in use in quantum register '{qreg_label}'.")

        self.qregisters[qreg_label].add(qubit or Qubit(str(self.num_qubits)))

        # remove cached 'qubits' attribute when updating 'qregisters';
        # will always be in 'self.__dict__' since 'self.qubits' is called above
        if "qubits" in self.__dict__:  # pragma: no cover
            del self.qubits

    def add_bit(self, bit: Optional[Bit] = None, creg_label: Optional[Hashable] = None) -> None:
        """Add a single bit to a classical register.

        Args:
            bit: Bit to add to the circuit. If ``None``, then a new bit is created.
            creg_label: Label for the classical register to which the new bit should be
                appended (defaults to 'r' followed by a random integer ID number).
        """
        if bit is not None and not isinstance(bit, Bit):
            raise TypeError(f"Can only add bits to circuit. Got '{type(bit).__name__}'.")

        if creg_label is not None:
            if creg_label not in self.cregisters:
                self.add_cregister(label=creg_label)
        else:
            if not self.cregisters:
                self.add_cregister()
            creg_label = list(self.cregisters)[0]

        # NOTE: same bit in different registers NOT allowed
        if self.bits and bit in self.bits:
            raise ValueError(f"Bit '{bit}' already in use in classical register '{creg_label}'.")

        self.cregisters[creg_label].add(bit or Bit(str(self.num_bits)))

        # remove cached 'bits' attribute when updating 'cregisters'
        # will always be in 'self.__dict__' since 'self.bits' is called above
        if "bits" in self.__dict__:  # pragma: no cover
            del self.bits

    def add_qregister(self, num_qubits: int = 0, label: Hashable = None) -> None:
        """Adds a new quantum register to the circuit.

        Args:
            num_qubits: Number of qubits in the quantum register (defaults to 0, i.e., empty).
            label: Quantum register label (defaults to 'qreg' followed by a incrementing
                integer starting at 0).
        """
        if label is None:
            label = f"qreg{len(self.qregisters)}"

        if label in self._qregisters:
            raise ValueError(f"Quantum register {label} already present in the circuit.")

        data = (Qubit(str(i)) for i in range(self.num_qubits, self.num_qubits + num_qubits))
        self._qregisters[label] = QuantumRegister(data)

        # remove cached 'qubits' attribute when updating 'qregisters'
        # will always be in 'self.__dict__' since 'self.qubits' is called above
        if "qubits" in self.__dict__:  # pragma: no cover
            del self.qubits

    def add_cregister(self, num_bits: int = 0, label: Hashable = None) -> None:
        """Adds a new classical register to the circuit.

        Args:
            num_qubits: Number of bits in the classical register (defaults to 0, i.e., empty).
            label: Classical register label (defaults to 'creg' followed by a incrementing
                integer starting at 0).
        """
        if label is None:
            label = f"creg{len(self.cregisters)}"

        if label in self._cregisters:
            raise ValueError(f"Classical register {label} already present in the circuit")

        data = (Bit(str(i)) for i in range(self.num_bits, self.num_bits + num_bits))
        self._cregisters[label] = ClassicalRegister(data)

        # remove cached 'bits' attribute when updating 'cregisters'
        # will always be in 'self.__dict__' since 'self.bits' is called above
        if "bits" in self.__dict__:  # pragma: no cover
            del self.bits

    def __repr__(self) -> str:
        """Returns the representation of the Circuit object."""
        qb, cb = len(self.qubits), len(self.bits)
        return f"<{self.__class__.__name__}: qubits={qb}, bits={cb}, ops={len(self.circuit)}>"

    def get_qubit(
        self, label: str, qreg_label: Optional[str] = None, return_all: bool = False
    ) -> Qubits:
        """Returns the Qubit(s) with a specific label.

        Args:
            label: The label of the qubit.
            qreg_label: Label of the quantum register to search in. If ``None``, all registers
                are searched.
            return_all: Whether to return all qubits if several exists with the same label,
                or just return the first occuring qubit. Defaults to ``False``.

        Returns:
            Qubit: Qubit with the specified label. If there are several with the same name,
            only the first occuring qubit will be returned unless ``return_all`` is ``True``.
        """
        if qreg_label:
            qregisters = self.qregisters[qreg_label]
        else:
            qregisters = itertools.chain.from_iterable(self.qregisters.values())

        qubits = []
        for qubit in qregisters:
            if qubit.label == label:
                if not return_all:
                    return qubit
                qubits.append(qubit)
        if qubits:
            return qubits

        raise ValueError(f"Qubit '{label}' not found.")

    def find_qubit(self, qubit: Qubit, qreg_label: bool = False) -> Tuple[Hashable, int]:
        """Returns the register where a qubit contained and its index in the register.

        Args:
            qubit: The qubit to find.
            qreg_label: Whether to return the containing quantum register label (``True``)
                or its index (``False``) in the ``self.qregisters`` dictionary.

        Returns:
            tuple: Tuple containing the quantum register label and the index of the qubit in that
            register.

        Raises:
            ValueError: If the qubit is not found in any register.
        """
        for i, (label, qreg) in enumerate(self.qregisters.items()):
            if qubit in qreg:
                idx = qreg.index(qubit)
                if qreg_label:
                    return (label, idx)
                return (i, idx)

        raise ValueError(f"Qubit {qubit} not found in any register.")

    def get_bit(
        self, label: str, creg_label: Optional[str] = None, return_all: bool = False
    ) -> Bits:
        """Returns the Bit(s) with a specific label.

        Args:
            label: The label of the bit.
            creg_label: Label of the classical register to search in. If ``None``, all registers
                are searched.
            return_all: Whether to return all bits if several exists with the same label,
                or just return the first occuring bit. Defaults to ``False``.

        Returns:
            Bit: Bit with the specified label. If there are several with the same name,
            only the first occuring bit will be returned unless ``return_all`` is ``True``.
        """
        if creg_label:
            cregisters = self.cregisters[creg_label]
        else:
            cregisters = itertools.chain.from_iterable(self.cregisters.values())

        bits = []
        for bit in cregisters:
            if bit.label == label:
                if not return_all:
                    return bit
                bits.append(bit)
        if bits:
            return bits

        raise ValueError(f"Bit '{label}' not found.")

    def find_bit(self, bit: Bit, creg_label: bool = False) -> Tuple[Hashable, int]:
        """Returns the register where a bit contained and its index in the register.

        Args:
            bit: The bit to find.
            creg_label: Whether to return the containing classical register label (``True``)
                or its index (``False``) in the ``self.cregisters`` dictionary.

        Returns:
            tuple: Tuple containing the classical register label and the index of the bit in that
            register.

        Raises:
            ValueError: If the bit is not found in any register.
        """
        for i, (label, creg) in enumerate(self.cregisters.items()):
            if bit in creg:
                idx = creg.index(bit)
                if creg_label:
                    return (label, idx)
                return (i, idx)

        raise ValueError(f"Bit {bit} not found in any register.")

    def to_qasm(self, version: str = "2.0", **kwargs) -> str:
        """Converts the Circuit into an OpenQASM string.

        Args:
            version: OpenQASM version (currently only supports 2.0).

        Keyword args:
            reg_labels: Whether to use the qregister labels (``True``) given or to simplify them
                and simply use standard OpenQASM 2.0 labels, ``q0``, ``q1``, etc., instead
                (``False``). Defaults to ``False``.
            gate_definitions: Whether to add definitions for gates that are not part of the
                OpenQASM standard library file ``qelib1.inc``. Defaults to ``False``.

        Returns:
            str: OpenQASM string representation of the circuit.
        """
        if version != "2.0":
            raise NotImplementedError("Only OpenQASM 2.0 is supported at the moment.")

        reg_labels = kwargs.get("reg_labels", False)
        gate_definitions = kwargs.get("gate_definitions", False)

        header_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

        qasm_str = ""
        # add quantum registers (only add index if more than one register)
        if len(self.qregisters) == 1:
            label, qreg = list(self.qregisters.items())[0]
            qasm_str += qreg.to_qasm(label=label if reg_labels else None) + ";\n"
        else:
            for i, (label, qreg) in enumerate(self.qregisters.items()):
                qasm_str += qreg.to_qasm(label=label if reg_labels else None, idx=i) + ";\n"

        # add classical registers (only add index if more than one register)
        if len(self.cregisters) == 1:
            label, creg = list(self.cregisters.items())[0]
            qasm_str += creg.to_qasm(label=label if reg_labels else None) + ";\n"
        else:
            for i, (label, creg) in enumerate(self.cregisters.items()):
                qasm_str += creg.to_qasm(label=label if reg_labels else None, idx=i) + ";\n"

        # add blank line between register declarations and gates
        qasm_str += "\n"

        mapping = {}
        for qb in self.qubits:
            label, idx = self.find_qubit(qb, qreg_label=reg_labels)
            if not reg_labels:
                label = f"q{label}" if len(self.qregisters) != 1 else "q"
            mapping[qb] = label, idx

        for op in self.circuit:
            if gate_definitions and hasattr(op, "_qasm_decl"):
                header_str += "\n" + getattr(op, "_qasm_decl") + "\n"
            qasm_str += op.to_qasm(mapping) + ";\n"

        return header_str.strip() + "\n\n" + qasm_str.strip()

    def to_qir(self, add_external: bool = False, bitcode: bool = False) -> Union[str, bytes]:
        """Compile a circuit into QIR.

        Args:
            add_external: Whether to add external functions (not defined in QIR). If ``False``,
                functions marked as external will be decomposed into valid operations (if possible).
            bitcode: Whether to return QIR as a legible string or as bitcode.

        Returns:
            Union[str, bytes]: The QIR representation of the circuit.
        """
        # import outside toplevel to avoid circular imports
        from dwave.gate.qir.compiler import qir_module

        module = qir_module(self, add_external=add_external)
        module.compile()

        if bitcode:
            return module.bitcode
        return module.qir

    @classmethod
    def from_qir(cls, qir: Union[str, bytes], bitcode: bool = False) -> Circuit:
        """Load a circuit from a QIR string or bitcode.

        Args:
            qir: The QIR string or bitcode to load into a circuit.
            bitcode: Whether to expect the QIR as a string or as bitcode.

        Returns:
            Circuit: The circuit represented by the QIR string or bitcode.
        """
        # import outside toplevel to avoid circular imports
        from dwave.gate.qir.loader import load_qir_bitcode, load_qir_string

        if bitcode:
            return load_qir_bitcode(qir, cls())
        return load_qir_string(qir, cls())


class ParametricCircuit(Circuit):
    """Class to build and manipulate parametric quantum circuits.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_bits: Number of classical bits in the circuit.
    """

    def __init__(self, num_qubits: Optional[int] = None, num_bits: Optional[int] = None) -> None:
        self._parameter_register = SelfIncrementingRegister()

        super().__init__(num_qubits, num_bits)

    def __call__(
        self, parameters: List[complex], qubits: Qubits, bits: Optional[Bits] = None
    ) -> None:
        """Apply all the operations in the circuit within a circuit context.

        Args:
            parameters: Parameters to apply to any parametric gates.
            qubits: Qubits on which the circuit operations should be applied. The qubits used in
                the circuit will be exchanged with the corresponding ones (e.g., with the same
                index as) in the active context.
            bits: Bits in which to store measurement values if measurements have been made.

        Raises:
            ValueError: If an invalid number of qubits are passed.
            CircuitError: If called outside of an active context.
        """
        if CircuitContext.active_context:
            for i, var in enumerate(self._parameter_register):
                var.set(parameters[i])

        # delay reset variables function call to on context exit
        CircuitContext.on_exit_functions.append(self.reset_variables)

        return super().__call__(qubits=qubits, bits=bits)

    def unlock(self) -> None:
        """Unlocks the circuit allowing for further operations to be applied."""
        self._parameter_register._frozen = False
        return super().unlock()

    def eval(
        self, parameters: Optional[Sequence[Sequence[complex]]] = None, inplace: bool = False
    ) -> ParametricCircuit:
        """Evaluate circuit operations with explicit parameters.

        Args:
            parameters: Parameters to replace operation variables with. Overrides potential variable
                values. If ``None`` then variable values are used (if existent).
            inplace: Whether to evaluate the parameters on ``self`` or on a copy
                of ``self`` (returned).

        Returns:
            ParametricOperation: Either ``self`` or a copy of ``self``.

        Raises:
            ValueError: If no parameters are passed and if variable has no set value.
        """
        circuit = self if inplace else copy.deepcopy(self)

        for i, op in enumerate(circuit.circuit):
            eval = getattr(op, "eval", None)
            params = parameters[i] if parameters else None
            circuit.circuit[i] = eval(params, inplace) if eval else op

        return circuit

    def reset_variables(self) -> None:
        """Resets any variables in the parameter register by setting their values to ``None``."""
        for variable in self._parameter_register:
            variable.reset()  # type: ignore

    @property
    def parametric(self) -> bool:
        """Whether the circuit has parameter variables."""
        return bool(self._parameter_register)

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the circuit."""
        return len(self._parameter_register)

    @property
    def context(self) -> ParametricCircuitContext:
        """Circuit context used to apply operations to the circuit."""
        if self._circuit_context is None:
            self._circuit_context = ParametricCircuitContext(circuit=self)

        assert isinstance(self._circuit_context, ParametricCircuitContext)
        return self._circuit_context

    def to_qasm(self) -> str:
        """Converts the Circuit into an OpenQASM string.

        Note, not supported for parametric circuits.
        """
        raise CircuitError("Parametric circuits cannot be transpiled into OpenQASM.")


Registers = NamedTuple("Registers", [("q", QuantumRegister), ("c", ClassicalRegister)])
"""NamedTuple: collection of registers returned from the context manager."""


class CircuitContext:
    """Class used to handle and store the active context.

    Args:
        circuit: Circuit to which the context is attached
    """

    _active_context: Optional[CircuitContext] = None
    """Current active context; can only be one at a time during runtime."""
    on_exit_functions: List[Callable] = []
    """List of functions that should be called on context exit. Cleared on context exit."""

    def __init__(self, circuit: Circuit) -> None:
        self._circuit = circuit
        self._frozen = False

    @property
    def circuit(self) -> Circuit:
        """Circuit attached to the context."""
        return self._circuit

    @property
    def frozen(self) -> bool:
        "Whether the context is frozen and no operations can be appended."
        return self._frozen

    @mixedproperty
    def freeze(cls) -> ContextManager:
        """Freeze the context so that no operations are appended on initialization.

        Returns a context manager for a context in which any initialized gates won't be appended to
        the active circuit context.

        Returns:
            ContextManager: Manager for context withing no opperations are appended.

        Raises:
            CircuitError: If used outside of a circuit context.

        Example:

            .. code-block:: python

                >>> from dwave.gate import Circuit
                >>> from dwave.gate.operations import X, Y, Z

                >>> circuit = Circuit(1)

                >>> with circuit.context as (q, c):
                ...   X(q[0])  # will be appended to the circuit
                ...   with circuit.context.freeze:
                ...       Y(q[0])  # will NOT be appended to the circuit
                ...   Z(q[0])  # will be appended to the circuit

                >>> print(circuit)
                <Operation: X, qubits=('q0',)>
                <Operation: Z, qubits=('q0',)>
        """

        class FrozenContext:
            def __enter__(self) -> None:
                if cls.active_context is None:
                    raise CircuitError("Can only freeze active context. No active context found.")
                cls.active_context._frozen = True

            def __exit__(self, _, __, ___) -> None:
                cls.active_context._frozen = False

        return FrozenContext()

    def __enter__(
        self,
    ) -> Registers:
        """Enters the context and sets itself as active."""
        if self.circuit.is_locked() is True:
            raise CircuitError(
                "Circuit is locked and no more operations can be appended. To "
                "unlock the circuit, call 'Circuit.unlock()' first."
            )

        if self.active_context is None:
            CircuitContext._active_context = self
        else:
            raise RuntimeError("Cannot enter context, another circuit context is already active.")

        return Registers(self.circuit.qubits, self.circuit.bits)

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exits the context and locks the circuit."""
        # IDEA: add setting to automatically decompose qubits on exit
        # TODO: add setting to automatically add missing qubits on exit (or raise error)
        CircuitContext._active_context = None

        for exit_func in self.on_exit_functions:
            exit_func()
        self.on_exit_functions.clear()

        self.circuit.lock()

    @mixedproperty
    def active_context(cls) -> Optional[CircuitContext]:
        """Current active context (usually ``self``)."""
        return cls._active_context


ParametricRegisters = NamedTuple(
    "ParametricRegisters",
    [("p", SelfIncrementingRegister), ("q", QuantumRegister), ("c", ClassicalRegister)],
)
"""NamedTuple: collection of registers returned from the parametric context manager."""


class ParametricCircuitContext(CircuitContext):
    """Class used to handle and store the active context with parametric circuits.

    Args:
        circuit: Parametric circuit to which the context is attached.
    """

    def __init__(self, circuit: ParametricCircuit) -> None:
        if not isinstance(circuit, ParametricCircuit):
            raise TypeError("'ParametricCircuitContext' only works with 'ParametricCircuit'")

        super().__init__(circuit)

    def __enter__(
        self,
    ) -> ParametricRegisters:
        """Enters the context and sets itself as active."""
        # should always be a 'ParametricCircuit'; check in '__init__'
        assert isinstance(self.circuit, ParametricCircuit)

        qreg, creg = super().__enter__()

        return ParametricRegisters(self.circuit._parameter_register, qreg, creg)

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exits the context and locks the circuit."""
        # should always be a 'ParametricCircuit'; check in '__init__'
        assert isinstance(self.circuit, ParametricCircuit)

        self.circuit._parameter_register.freeze()
        super().__exit__(type, value, traceback)
