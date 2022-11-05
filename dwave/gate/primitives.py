from typing import Hashable

from dwave.gate.tools.counters import IDCounter


class Qubit:
    """Qubit type.

    Args:
        label: Label used to represent the qubit.
    """

    def __init__(self, label: Hashable) -> None:
        self._label = label
        self._id: str = IDCounter.next()

    @property
    def label(self) -> Hashable:
        """The qubit's name."""
        return self._label

    @property
    def id(self) -> str:
        """The qubit's unique identification number."""
        return self._id

    def __eq__(self, __o: object) -> bool:
        """Two qubits are equal if they share the same id."""
        if isinstance(__o, Qubit):
            return self.id == __o.id
        return False

    def __repr__(self) -> str:
        """The representation of the variable is its label."""
        return f"<qubit: {self.label}, id:{self.id}>"

    def __hash__(self) -> int:
        """The hash of the qubit is determined by its id."""
        return hash(self.__class__.__name__ + self.id)


class Bit:
    """Classical bit type.

    Args:
        label: Label used to represent the bit.
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._id: str = IDCounter.next()

    @property
    def label(self) -> str:
        """The bit's label."""
        return self._label

    @property
    def id(self) -> str:
        """The bit's unique identification number."""
        return self._id

    def __eq__(self, __o: object) -> bool:
        """Two bits are equal if they share the same id."""
        if isinstance(__o, Bit):
            return self.id == __o.id
        return False

    def __repr__(self) -> str:
        """The representation of the variable is its label."""
        return f"<bit: {self.label}, id:{self.id}>"

    def __hash__(self) -> int:
        """The hash of the qubit is determined by its id."""
        return hash(self.__class__.__name__ + self.id)


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
        """The variable name."""
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
