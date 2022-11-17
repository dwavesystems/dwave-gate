# Confidential & Proprietary Information: D-Wave Systems Inc.
from __future__ import annotations

__all__ = [
    "CircuitError",
    "Circuit",
    "ParametricCircuit",
    "CircuitContext",
    "ParametricCircuitContext",
]

from types import TracebackType
from typing import (
    TYPE_CHECKING,
    ContextManager,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from dwave.gate.mixedproperty import mixedproperty
from dwave.gate.primitives import Bit, Qubit
from dwave.gate.registers import (
    ClassicalRegister,
    QuantumRegister,
    SelfIncrementingRegister,
    Variable,
)

if TYPE_CHECKING:
    from dwave.gate.operations.base import Operation


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

        self._locked = False

    def __call__(self, *args, **kwargs) -> None:
        """Apply all the operations in the circuit within a circuit context.

        If passed to the call method, ``parameters`` is assumed to be the first argument (unless
        passed as a keyword argument). If only qubits are passed (required since a circuit can only
        be applied within a context), ``qubits`` will be the first argument (unless passed as a
        keyword argument).

        Args:
            parameters (optional): Parameters to apply to any parametric gates.
            qubits: Qubits on which thecircuit operations should be applied. The qubits used in
                the circuit will be exchanged with the corresponding ones (e.g., with the same
                index as) in the active context.

        Raises:
            ValueError: If the wrong number of qubits is passed.
            CircuitError: If called outside of an active context.
            TypeError: If an unexpected keyword argument is passed, or if an invalid number of
                total arguments is passed (expects 1-2 arguments).
        """

        qubits, _ = self._extract_arguments(*args, **kwargs)

        if CircuitContext.active_context is None:
            raise CircuitError("Can only apply circuit object inside a circuit context.")
        if CircuitContext.active_context.circuit is self:
            raise TypeError("Cannot apply circuit in its own context.")

        qubit_map = dict(zip(self.qubits, qubits))
        for op in self.circuit:
            mapped_qubits = [qubit_map[qb] for qb in op.qubits or []]

            # NOTE: avoid circular imports; needed to check operation type
            from dwave.gate.operations.base import ControlledOperation, ParametricOperation

            if isinstance(op, ParametricOperation):
                for i, p in enumerate(op.parameters):
                    if isinstance(p, Variable):
                        op.parameters[i] = kwargs["parameters"].pop(0)
                op.__class__(op.parameters, qubits=mapped_qubits)
            elif isinstance(op, ControlledOperation):
                op.__class__(*mapped_qubits)
            else:
                op.__class__(qubits=mapped_qubits)

    def _extract_arguments(
        self, *args, **kwargs
    ) -> Tuple[Sequence[Qubit], Optional[List[complex]]]:
        """Extracts the correct parameter and qubit arrays from args and kwargs.

        Looks for "parameters" and "qubits" in the arguments. Raises errors if unexpected keyword
        arguments are passed or an invalid number of arguments are passed.
        """
        # assert that args and/or kwargs only contain qubits and/or parameters
        invalid_kwargs = set(kwargs) - {"qubits", "parameters"}
        if invalid_kwargs:
            raise TypeError(f"__call__() got unexpected keyword argument {invalid_kwargs.pop()}")

        if len(args) + len(kwargs) > 2:
            raise TypeError(
                f"__call__() takes from 1 to 2 arguments but {len(args) + len(kwargs)} were given"
            )

        if args or kwargs:
            qubits = kwargs.get("qubits", None) or args[0]
        else:
            qubits = []

        if isinstance(qubits, Qubit):
            qubits = [qubits]

        if len(qubits) != len(self.qubits):
            raise ValueError(f"Circuit requires {len(self.qubits)} qubits, got {len(qubits)}.")

        return qubits, kwargs.get("parameters", None)

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

    def append(self, operation: Union[Sequence[Operation], Operation]) -> None:
        """Appends an operation to the circuit.

        Args:
            operation: Operation or sequence of operations to append to the circuit.
        """
        if self.is_locked() == True:
            raise CircuitError(
                "Circuit is locked and no more operations can be appended. To "
                "unlock the circuit, call 'Circuit.unlock()' first."
            )

        if not isinstance(operation, Sequence):
            operation = [operation]

        for op in operation:
            for q in op.qubits or []:
                if q not in self.qubits:
                    raise ValueError(f"Qubit '{q}' not in circuit.")

        self._circuit.extend(operation)

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

    @property
    def qubits(self) -> Sequence[Qubit]:
        """Qubits handled by the circuit."""
        qubits = [qb for qreg in self.qregisters.values() for qb in qreg]
        return qubits

    @property
    def bits(self) -> Sequence[Bit]:
        """Classical bits handled by the circuit."""
        bits = [b for creg in self.cregisters.values() for b in creg]
        return bits

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
        """Circuit context used to apply operations to the circuit."""
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
        if qreg_label is not None:
            if qreg_label not in self.qregisters:
                self.add_qregister(label=qreg_label)
        else:
            if not self.qregisters:
                self.add_qregister()
            qreg_label = list(self.qregisters)[0]

        # NOTE: same qubit in different registers NOT allowed
        if qubit in self.qubits:
            raise ValueError(
                f"Qubit '{qubit}' already in use in quantum register '{self.qregisters[qreg_label].label}'."
            )

        self.qregisters[qreg_label].add(qubit or Qubit(str(self.num_qubits)))

    def add_bit(self, bit: Optional[Bit] = None, creg_label: Optional[Hashable] = None) -> None:
        """Add a single bit to a classical register.

        Args:
            bit: Bit to add to the circuit. If ``None``, then a new bit is created.
            creg_label: Label for the classical register to which the new bit should be
                appended (defaults to 'r' followed by a random integer ID number).
        """
        if creg_label is not None:
            if creg_label not in self.cregisters:
                self.add_cregister(label=creg_label)
        else:
            if not self.cregisters:
                self.add_cregister()
            creg_label = list(self.cregisters)[0]

        # NOTE: same bit in different registers NOT allowed
        if bit in self.bits:
            raise ValueError(f"Bit '{bit}' already in use in classical register '{creg_label}'.")

        self.cregisters[creg_label].add(bit or Bit(str(self.num_bits)))

    def add_qregister(self, num_qubits: int = 0, label: Hashable = None) -> None:
        """Adds a new quantum register to the circuit.

        Args:
            num_qubits: Number of qubits in the quantum register (defaults to 0, i.e., empty).
            label: Quantum register label (defaults to 'qreg' followed by a incrementing integer starting at 0).
        """
        if label is None:
            label = f"qreg{len(self.qregisters)}"

        if label in self._qregisters:
            raise ValueError(f"Quantum register {label} already present in the circuit.")

        data = [Qubit(str(i)) for i in range(num_qubits)]
        qreg = QuantumRegister(label=label, data=data)

        self._qregisters[label] = qreg

    def add_cregister(self, num_bits: int = 0, label: Hashable = None) -> None:
        """Adds a new classical register to the circuit.

        Args:
            num_qubits: Number of bits in the classical register (defaults to 0, i.e., empty).
            label: Classical register label (defaults to 'creg' followed by a incrementing integer starting at 0).
        """
        if label is None:
            label = f"creg{len(self.cregisters)}"

        if label in self._cregisters:
            raise ValueError(f"Classical register {label} already present in the circuit")

        data = [Bit(str(i)) for i in range(num_bits)]
        creg = ClassicalRegister(label=label, data=data)

        self._cregisters[label] = creg

    def __repr__(self) -> str:
        """Returns the representation of the Circuit object."""
        qb, cb = len(self.qubits), len(self.bits)
        return f"<Circuit: qubits={qb}, bits={cb}, ops={len(self.circuit)}>"

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
            qasm_str += list(self.qregisters.values())[0].to_qasm(qreg_label=reg_labels) + ";\n"
        else:
            for i, (_, qreg) in enumerate(self.qregisters.items()):
                qasm_str += qreg.to_qasm(qreg_label=reg_labels, idx=i) + ";\n"

        # add classical registers (only add index if more than one register)
        if len(self.cregisters) == 1:
            qasm_str += list(self.cregisters.values())[0].to_qasm(creg_label=reg_labels) + ";\n"
        else:
            for i, (_, creg) in enumerate(self.cregisters.items()):
                qasm_str += creg.to_qasm(creg_label=reg_labels, idx=i) + ";\n"

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


class ParametricCircuit(Circuit):
    """Class to build and manipulate parametric quantum circuits.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_bits: Number of classical bits in the circuit.
    """

    def __init__(self, num_qubits: Optional[int] = None, num_bits: Optional[int] = None) -> None:
        self._parameter_register = SelfIncrementingRegister("preg")

        super().__init__(num_qubits, num_bits)

    def __call__(self, *args, **kwargs) -> None:
        if "parameters" in kwargs:
            parameters = kwargs.pop("parameters")
        else:
            parameters = args[0]
            args = args[1:]

        return super().__call__(*args, **kwargs, parameters=parameters)

    def unlock(self) -> None:
        """Unlocks the circuit allowing for further operations to be applied."""
        self._parameter_register._frozen = False
        return super().unlock()

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


class CircuitContext:
    """Class used to handle and store the active context.

    Args:
        circuit: Circuit to which the context is attached
    """

    _active_context = None
    """Optional[CircuitContext]: Current active context; can only be one at a time during runtime."""

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

        Returns a context manager for a context in which any initialized gates won't be
        appended to the active circuit context.

        Returns:
            ContextManager: Manager for context withing no opperations are appended.

        Raises:
            CircuitError: If used outside of a circuit context.

        Example:

            .. code-block:: python

                >>> from dwave.gate import Circuit
                >>> from dwave.gate.operations import X, Y, Z

                >>> circuit = Circuit(1)

                >>> with circuit.context as q:
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
    ) -> Sequence[Qubit]:
        """Enters the context and sets itself as active."""
        if self.circuit.is_locked() == True:
            raise CircuitError(
                "Circuit is locked and no more operations can be appended. To "
                "unlock the circuit, call 'Circuit.unlock()' first."
            )

        if self.active_context is None:
            CircuitContext._active_context = self
        else:
            raise RuntimeError("Cannot enter context, another circuit context is already active.")

        return self.circuit.qubits

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
        self.circuit.lock()

    @mixedproperty
    def active_context(cls) -> Optional[CircuitContext]:
        """Current active context (usually ``self``)."""
        return cls._active_context


class ParametricCircuitContext(CircuitContext):
    """Class used to handle and store the active context with parametric circuits.

    Args:
        circuit: Parametric circuit to which the context is attached.
    """

    def __init__(self, circuit: Circuit) -> None:
        if not isinstance(circuit, ParametricCircuit):
            raise TypeError("'ParametricCircuitContext' only works with 'ParametricCircuit'")

        super().__init__(circuit)

    def __enter__(
        self,
    ) -> Tuple[SelfIncrementingRegister, Sequence[Qubit]]:
        """Enters the context and sets itself as active."""
        # should always be a 'ParametricCircuit'; check in '__init__'
        assert isinstance(self.circuit, ParametricCircuit)

        q = super().__enter__()
        return (self.circuit._parameter_register, q)

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
