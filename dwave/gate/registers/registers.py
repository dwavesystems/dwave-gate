# Copyright 2022 D-Wave Systems Inc.
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

from __future__ import annotations

__all__ = [
    "RegisterError",
    "Register",
    "QuantumRegister",
    "ClassicalRegister",
    "SelfIncrementingRegister",
]

from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Hashable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from dwave.gate.primitives import Bit, Qubit, Variable
from dwave.gate.registers.cyregister import cyRegister

if TYPE_CHECKING:
    from typing_extensions import Self

Data = TypeVar("Data", bound=Hashable)


class RegisterError(Exception):
    """Exception to be raised when there is an error with a Register."""


class Register(cyRegister, AbstractSet[Data], Sequence[Data]):
    """Register to store qubits and/or classical bits.

    Args:
        data: Sequence of hashable data items (defaults to empty).
    """

    def __init__(self, data: Optional[Sequence[Data]] = None) -> None:
        self._frozen = False
        super().__init__(data or [])

    def __getitem__(self, idx) -> Data:
        """Get item at index."""
        return super().__getitem__(idx)  # type: ignore

    def __iter__(self) -> Iterator[Data]:
        """Return a data iterator."""
        return super().__iter__()  # type: ignore

    @property
    def data(self) -> Sequence[Data]:
        """Sequence of qubits or bits."""
        return list(self)

    @property
    def frozen(self) -> bool:
        """Whether the register is frozen and no more data can be added."""
        return self._frozen

    def freeze(self) -> None:
        """Freezes the register so that no (qu)bits can be added or removed."""
        self._frozen = True

    def __str__(self) -> str:
        """Returns the data as a string."""
        return f"{self.__class__.__name__}({list(self)})"

    def __repr__(self) -> str:
        """Returns the representation of the Register object."""
        return f"<{self.__class__.__name__}, data={list(self)}>"

    def __add__(self, qreg: Register) -> Self:
        """Adds two registers together and returns a register referencing the contained items.

        Args:
            qreg: Register to add to ``self``.

        Returns:
            Register: Combined register referencing the contained items.
        """
        copy = self.copy()
        copy._extend(qreg)
        return copy

    def add(self, data: Union[Hashable, Sequence[Hashable], Register]) -> None:
        """Add one or more data items to the register.

        Args:
            data: One or more data items to add to the register.
        """
        if self.frozen:
            raise RegisterError("Register is frozen and no more data can be added.")

        if not isinstance(data, Register) and (
            isinstance(data, str) or not isinstance(data, Sequence)
        ):
            data = [data]

        self._extend(data)


class QuantumRegister(Register[Qubit]):
    """Quantum register to store qubits.

    Args:
        data: Sequence of qubits (defaults to empty).
    """

    def __init__(self, data: Optional[Sequence[Qubit]] = None) -> None:
        super().__init__(data)

    def to_qasm(self, label: Hashable = None, idx: Optional[int] = None) -> str:
        """Converts the quantum register into an OpenQASM string.

        Args:
            label: Optional label for the quantum register.
            idx: Optional index number for quantum register.

        Returns:
            str: OpenQASM string representation of the circuit.
        """
        if label:
            return f"qreg {str(label)}[{len(self)}]"

        idx_str = str(idx) if idx is not None else ""
        return f"qreg q{idx_str}[{len(self)}]"

    def freeze(self) -> None:
        """Freezes the register so that no qubits can be added or removed."""
        return super().freeze()


class ClassicalRegister(Register[Bit]):
    """Classical register to store bits.

    Args:
        data: Sequence of bits (defaults to empty).
    """

    def __init__(self, data: Optional[Sequence[Bit]] = None) -> None:
        super().__init__(data)

    def to_qasm(self, label: Hashable = None, idx: Optional[int] = None) -> str:
        """Converts the classical register into an OpenQASM string.

        Args:
            label: Optional label for the quantum register.
            idx: Optional index number for quantum register.

        Returns:
            str: OpenQASM string representation of the circuit.
        """
        if label:
            return f"creg {str(label)}[{len(self)}]"

        idx_str = str(idx) if idx is not None else ""
        return f"creg c{idx_str}[{len(self)}]"

    def freeze(self) -> None:
        """Freezes the register so that no bits can be added or removed."""
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

    def __init__(self, data: Optional[Sequence[Variable]] = None) -> None:
        super().__init__(data)

    def freeze(self) -> None:
        """Freezes the register so that no data can be added or removed."""
        return super().freeze()

    def __getitem__(self, index: int) -> Variable:
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
            self.add([Variable(str(i)) for i in range(len(self.data), index + 1)])
        return super().__getitem__(index)  # type: ignore
