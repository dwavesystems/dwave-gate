# Confidential & Proprietary Information: D-Wave Systems Inc.
from abc import abstractmethod
from collections.abc import Collection
from typing import Hashable, Iterator, Optional, Sequence, Union


class Register(Collection):
    """Register to store qubits and/or classical bits.

    Args:
        label: Quantum or classical register label.
        data: Sequence of qubits or bits (defaults to empty).
    """

    def __init__(
        self, label: Hashable, data: Optional[Sequence[Hashable]] = None
    ) -> None:
        self._label: str = label
        self._data: Sequence[Hashable] = data or []

    @property
    def label(self) -> Hashable:
        """Quantum or classical register label."""
        return self._label

    @property
    def data(self) -> Sequence[Hashable]:
        """Sequence of qubits or bits."""
        return self._data

    @abstractmethod
    def freeze(self) -> None:
        """Freezes the register so that no (qu)bits can be added or removed."""
        raise NotImplementedError("Freezing the register is not currently supported.")

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

    def add(self, labels: Union[Hashable, Sequence[Hashable]]) -> None:
        """Add one or more (qu)bits to the register.

        Args:
            labels: One or more (qu)bit labels to add to the register.
        """
        if isinstance(labels, str) or not isinstance(labels, Sequence):
            labels = [labels]

        duplicate_labels = set(labels).intersection(self._data)
        if len(duplicate_labels) != 0:
            raise ValueError(f"Label(s) '{duplicate_labels}' already in use")
        self._data.extend(labels)


class QuantumRegister(Register):
    """Quantum register to store qubits..

    Args:
        label: Quantum register label.
        data: Sequence of qubits (defaults to empty).
    """

    def __init__(
        self, label: Hashable, data: Optional[Sequence[Hashable]] = None
    ) -> None:
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
    """Classical register to store qubits..

    Args:
        label: Classical register label.
        data: Sequence of bits (defaults to empty).
    """

    def __init__(
        self, label: Hashable, data: Optional[Sequence[Hashable]] = None
    ) -> None:
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
