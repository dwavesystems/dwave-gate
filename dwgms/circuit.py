# Confidential & Proprietary Information: D-Wave Systems Inc.
from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    ContextManager,
    Dict,
    Hashable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from dwgms.mixedproperty import mixedproperty
from dwgms.registers import ClassicalRegister, QuantumRegister, SelfIncrementingRegister, Variable

if TYPE_CHECKING:
    from dwgms.operations.base import Operation

# IDEA: support same qubit/bit label in multiple registers,
# prefixing them with the register name e.g., `r192548::my_qubit`


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
        parametric: bool = False,
    ) -> None:
        self._circuit: Sequence[Operation] = []
        self._circuit_context: Optional[CircuitContext] = None

        # registers for quantum and classical bits
        self._qregisters: Dict[Hashable, QuantumRegister] = dict()
        self._cregisters: Dict[Hashable, ClassicalRegister] = dict()

        if num_qubits is not None:
            self.add_qregister(num_qubits=num_qubits)

        if num_bits is not None:
            self.add_cregister(num_bits=num_bits)

        self._locked = False

        self._parametric = parametric
        if parametric:
            self._parameter_register = SelfIncrementingRegister("preg")

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
        qubits, parameters = self._extract_arguments(*args, **kwargs)

        # TODO: update to check for single qubit instead of str
        if isinstance(qubits, str) or not isinstance(qubits, Sequence):
            qubits = tuple([qubits])

        if len(qubits) != len(self.qubits):
            raise ValueError(f"Circuit requires {len(self.qubits)} qubits, got {len(qubits)}.")

        if CircuitContext.active_context is None:
            raise CircuitError("Can only apply circuit object inside a circuit context.")

        qubit_map = dict(zip(self.qubits, qubits))
        for op in self.circuit:
            mapped_qubits = [qubit_map[qb] for qb in op.qubits]
            if hasattr(op, "parameters"):
                self._apply_parametric_ops(op, parameters, mapped_qubits)
            elif hasattr(op, "control"):
                op.__class__(*mapped_qubits)
            else:
                op.__class__(qubits=mapped_qubits)

    def _extract_arguments(self, *args, **kwargs) -> Tuple[Sequence[Hashable], Sequence[complex]]:
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

        # if parameters in 'args', will always be the first arg (similar to 'ParametricOperation')
        if (
            len(args) == 2 or (len(args) == 1 and "qubits" in kwargs)
        ) and "parameters" not in kwargs:
            parameters = args[0]
        elif len(args) <= 1 and "parameters" in kwargs:
            parameters = kwargs["parameters"]
        else:
            parameters = None

        # if qubits are in 'args', will always be last (even if lonely)
        qubits = kwargs.get("qubits", None) or args[-1]

        return qubits, parameters

    def _apply_parametric_ops(self, op, parameters, qubits):
        """Replaces variable instances with parameter values and applies the gate to the
        active circuit."""
        applied_params = op.parameters.copy()
        if parameters is not None:
            for i, p in enumerate(applied_params):
                if isinstance(p, Variable):
                    idx = self._parameter_register.index(p)
                    applied_params[i] = parameters[idx]

        op.__class__(*applied_params, qubits=qubits)

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
    def circuit(self) -> Sequence[Operation]:
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
            for q in op.qubits:
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
    def parametric(self) -> bool:
        """Whether the circuit has parameter variables."""
        return self._parametric

    @property
    def qubits(self) -> Sequence[Hashable]:
        """Qubits handled by the circuit."""
        qubits = [qb for qreg in self.qregisters.values() for qb in qreg]
        return qubits

    @property
    def bits(self) -> Sequence[Hashable]:
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
    def num_parameters(self) -> int:
        """Number of parameters in the circuit."""
        if self.parametric:
            return len(self._parameter_register)
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
        self._circuit = []
        self._circuit_context = None
        if not keep_registers:
            self._qregisters = None
            self._cregisters = None

        self.unlock()

    def add_qubit(self, label: Hashable = None, qreg_label: Optional[Hashable] = None) -> None:
        """Add a single qubit to a quantum register.

        Args:
            label: Label for the qubit (defaults to 'q' followed by an incrementing
                integer, e.g., 'q0', 'q1', 'q42').
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

        if label is None:
            label = "q" + str(self.num_qubits)

        # NOTE: duplicate qubit labels in different registers NOT allowed
        if label in self.qubits:
            raise ValueError(
                f"Qubit label '{label}' already in use in quantum register '{self.qregisters[qreg_label].label}'."
            )

        self.qregisters[qreg_label].add(label)

    def add_bit(self, label: Hashable = None, creg_label: Optional[Hashable] = None) -> None:
        """Add a single bit to a classical register.

        Args:
            label: Label for the bit (defaults to 'c' followed by an incrementing
                integer, e.g., 'c0', 'c1', 'c42').
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

        if label is None:
            label = "c" + str(self.num_bits)

        # NOTE: duplicate qubit labels in different registers NOT allowed
        if label in self.bits:
            raise ValueError(
                f"Bit label '{label}' already in use in classical register '{creg_label}'."
            )

        self.cregisters[creg_label].add(label)

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

        data = ["q" + str(i) for i in range(num_qubits)]
        qreg = QuantumRegister(label=label, data=data)

        # TODO: freezing not currently supported
        # qreg.freeze()

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

        data = ["c" + str(i) for i in range(num_bits)]
        creg = ClassicalRegister(label=label, data=data)

        # TODO: freezing not currently supported
        # creg.freeze()

        self._cregisters[label] = creg

    def __str__(self) -> str:
        """Returns the circuit as an OpenQASM string."""
        return "\n".join([str(op) for op in self.circuit])

    def __repr__(self) -> str:
        """Returns the representation of the Circuit object."""
        qb, cb = len(self.qubits), len(self.bits)
        return f"<Circuit: qubits={qb}, bits={cb}, ops={len(self.circuit)}>"

    def to_qasm(self, version: str = "2.0") -> str:
        """Converts the Circuit into an OpenQASM string.

        Args:
            version: OpenQASM version (currently only supports 2.0).

        Returns:
            str: OpenQASM string representation of the circuit.
        """
        if version != "2.0":
            raise NotImplementedError("Only version 2.0 is supported at the moment.")

        qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        for l, q in self.qregisters.items():
            qasm_string += q.to_qasm()

        for l, c in self.cregisters.items():
            qasm_string += c.to_qasm()

        qasm_string += "\n"

        for i, ts in enumerate(self._time_slices):
            qasm_string += f"// time-slice {i}\n"
            for inst in ts.instructions:
                qasm_string += inst.to_qasm(qmapping=self.qubits, cmapping=self.bits)

        return qasm_string

    @classmethod
    def from_qasm(cls, path: Union[str, Path]) -> Circuit:
        """Converts an OpenQASM string into a Circuit object."""
        raise NotImplementedError(
            "Converting OpenQASM to Circuit object not supported at the moment."
        )


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

                >>> from dwgms import Circuit
                >>> from dwgms.operations import X, Y, Z

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
    ) -> Union[Sequence[Hashable], Tuple[SelfIncrementingRegister, Sequence[Hashable]]]:
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

        if self.circuit.parametric:
            return (self.circuit._parameter_register, self.circuit.qubits)
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
    def active_context(cls) -> CircuitContext:
        """Current active context (usually ``self``)."""
        return cls._active_context
