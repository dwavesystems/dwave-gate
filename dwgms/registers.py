# Confidential & Proprietary Information: D-Wave Systems Inc.
from collections.abc import Collection
from typing import Hashable, Iterator, Optional, Sequence, Union


class Variable:
    """Variable parameter type.

    Used as a placeholder for parameter values. Two variables with the same label are considered
    equal (``Variable('a') == Variable('a')``) and have the same hash value, but not identical
    (``Variable('a') is not Variable('a')``)

    Args:
        name: String used to represent the variable.
    """

    def __init__(self, name: str) -> None:
        self._name = str(name)

    @property
    def name(self) -> str:
        """The variable label."""
        return self._name

    def __eq__(self, __o: object) -> bool:
        """Two variables are equal if they share the same label."""
        if isinstance(__o, Variable):
            return self.name == __o.name
        return False

    def __repr__(self) -> str:
        """The representation of the variable is its label."""
        return f"{{{self.name}}}"

    def __hash__(self) -> int:
        """The hash of the variable is determined by its label."""
        return hash(self.name)


class RegisterError(Exception):
    """Exception to be raised when there is an error with a Register."""


class Register(Collection):
    """Register to store qubits and/or classical bits.

    Args:
        label: Quantum or classical register label.
        data: Sequence of qubits or bits (defaults to empty).
    """

    def __init__(
        self, label: Hashable, data: Optional[Union[Hashable, Sequence[Hashable]]] = None
    ) -> None:
        self._label: str = label
        self._data = []

        self._frozen = False

        # add the data via the add function to check for duplicates
        if data:
            self.add(data)

    @property
    def label(self) -> Hashable:
        """Quantum or classical register label."""
        return self._label

    @property
    def data(self) -> Sequence[Hashable]:
        """Sequence of qubits or bits."""
        return self._data

    @property
    def frozen(self) -> bool:
        """Whether the register is frozen and no more data can be added."""
        return self._frozen

    def freeze(self) -> None:
        """Freezes the register so that no (qu)bits can be added or removed."""
        self._frozen = True

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over the (qu)bits."""
        return self.data.__iter__()

    def __len__(self) -> int:
        """Return the length of the (qu)bit register."""
        return len(self.data)

    def __getitem__(self, index: int) -> Hashable:
        """Return the item at a specified index.

        Note that a ``Register`` differs from a ``Sequence`` by returning
        ``None`` instead of raising an ``IndexError`` when attemting to
        access an index outside of the collection.

        Args:
            Index of the (qu)bit to be returned.

        Returns:
            Hashable: The item at the specified index.
        """
        try:
            return self.data[index]
        except IndexError:
            return None

    def __contains__(self, item: Hashable) -> bool:
        """Check if an item is contained in the register.

        Args:
            item: Item to check if contained in register.

        Returns:
            bool: Whether the item is in the register.
        """
        return item in self.data

    def __str__(self) -> str:
        """Returns the data as a string."""
        return str(self.data)

    def __repr__(self) -> str:
        """Returns the representation of the Register object."""
        return f"<{self.__class__.__name__}, data={self.data}>"

    def index(self, item: object) -> Hashable:
        """Get the index of the passed item.

        Args:
            item: Object for which to get the index.

        Returns:
            Hashable: Index of the object in the register.
        """
        return self.data.index(item)

    def add(self, labels: Union[Hashable, Sequence[Hashable]]) -> None:
        """Add one or more data items to the register.

        Args:
            labels: One or more (qu)bit labels to add to the register.
        """
        if self.frozen:
            raise RegisterError("Register is frozen and no more data can be addded.")

        if isinstance(labels, str) or not isinstance(labels, Sequence):
            labels = [labels]

        duplicate_labels = set(labels).intersection(self._data)
        if len(duplicate_labels) != 0 or len(set(labels)) != len(labels):
            raise ValueError(f"Label(s) '{duplicate_labels}' already in use")
        self._data.extend(labels)


class QuantumRegister(Register):
    """Quantum register to store qubits.

    Args:
        label: Quantum register label.
        data: Sequence of qubits (defaults to empty).
    """

    def __init__(self, label: Hashable, data: Optional[Sequence[Hashable]] = None) -> None:
        super().__init__(label, data)

    def to_qasm(self) -> str:
        """Converts the quantum register into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the circuit.
        """
        return f"qreg {self._label}[{len(self)}]; \n"

    def freeze(self) -> None:
        """Freezes the register so that no (qu)bits can be added or removed."""
        return super().freeze()


class ClassicalRegister(Register):
    """Classical register to store qubits.

    Args:
        label: Classical register label.
        data: Sequence of bits (defaults to empty).
    """

    def __init__(self, label: Hashable, data: Optional[Sequence[Hashable]] = None) -> None:
        super().__init__(label, data)

    def to_qasm(self) -> str:
        """Converts the classical register into an OpenQASM string.

        Returns:
            str: OpenQASM string representation of the circuit.
        """
        return f"creg {self._label}[{len(self)}]; \n"

    def freeze(self) -> None:
        """Freezes the register so that no (qu)bits can be added or removed."""
        return super().freeze()


class SelfIncrementingRegister(Register):
    """Self-incrementing classical register to store parameter variables.

    The self-incrementing register will automatically add variable parameters when attemting to
    index outside of the register scope, and then return the requested index. For example,
    attempting to get parameter at index 3 in a ``SelfIncrementingRegister`` of length 1 would be
    equal to first running ``Register.add([Variable("p1"), Variable("p2")])`` and then returning
    ``Variable("p2")``.

    Args:
        label: Classical register label. data: Sequence of bits (defaults to empty).
        data: Sequence of parameters or variables (defaults to empty).
    """

    def __init__(self, label: Hashable, data: Optional[Sequence[Hashable]] = None) -> None:
        super().__init__(label, data)

    def freeze(self) -> None:
        """Freezes the register so that no data can be added or removed."""
        return super().freeze()

    def __getitem__(self, index: int) -> Hashable:
        """Return the parameter at a specified index.

        Warning, will _not_ raise an ``IndexError`` if attempting to access outside of the registers
        data sequence. It will instead automatically add variable parameters when attemting to index
        outside of the scope, and then return the newly created variable at the requested index.

        Args:
            Index of the parameter to be returned.

        Returns:
            Hashable: The parameter at the specified index.
        """
        if index >= len(self.data) and not self.frozen:
            self.add([Variable(i) for i in range(len(self.data), index + 1)])
        return self.data[index]
