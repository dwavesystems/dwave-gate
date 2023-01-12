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

import pytest

from dwave.gate.primitives import Bit, Qubit, Variable
from dwave.gate.registers.registers import (
    ClassicalRegister,
    QuantumRegister,
    RegisterError,
    SelfIncrementingRegister,
)


class TestQuantumRegister:
    """Unit tests for the quantum register class."""

    def test_empty_register(self):
        """Test initializing an empty data register."""
        reg = QuantumRegister()

        assert reg.data == []
        assert len(reg) == 0

    def test_register_with_data(self):
        """Test initializing an empty data register."""
        data = [Qubit(1), Qubit(2), Qubit(3), Qubit(42)]
        reg = QuantumRegister(data=data)

        assert reg.data == data
        assert len(reg) == 4

    def test_register_iteration(self):
        """Test iterating over a register."""
        data = [Qubit(1), Qubit(21), Qubit(32)]
        reg = QuantumRegister(data=data)

        for i, d in enumerate(reg):
            assert d == data[i]

    def test_index(self):
        """Test getting the index of an item in a register."""
        data = [Qubit(1), Qubit(13), Qubit(42), Qubit(54)]
        reg = QuantumRegister(data=data)

        assert reg.index(data[0]) == data.index(data[0])

    def test_register_indexing(self):
        """Test getting an item at a specific index in a register."""
        data = [Qubit(1), Qubit(13), Qubit(42), Qubit(54)]
        reg = QuantumRegister(data=data)

        # don't use enumerate so as to _only_ test indexing (and not iteration)
        for i in range(len(reg)):
            assert reg[i] == data[i]

    def test_register_indexing_out_of_range(self):
        """Test getting a item at a index out-of-range in a register."""
        reg = QuantumRegister(data=[Qubit(1), Qubit(13), Qubit(42), Qubit(54)])

        with pytest.raises(IndexError, match="index out of range"):
            reg[42]

    def test_contains(self):
        """Test checking if an item is in a register."""
        data = [Qubit(1), Qubit(13), Qubit(42), Qubit(54)]
        reg = QuantumRegister(data)

        assert data[1] in reg
        assert 657 not in reg
        assert Qubit(1) not in reg

    def test_repr(self):
        """Test the representation of a register."""
        data = [Qubit(1), Qubit(13), Qubit(42), Qubit(54)]
        reg = QuantumRegister(data)

        assert reg.__repr__() == f"<QuantumRegister, data={data}>"

    def test_str(self):
        """Test the string of a register."""
        data = [Qubit(1), Qubit(13), Qubit(42), Qubit(54)]
        reg = QuantumRegister(data)

        assert str(reg) == f"QuantumRegister({data})"

    def test_add(self):
        """Test adding qubits to the register."""
        data = [Qubit(0), Qubit(1)]
        reg = QuantumRegister(data=data)
        assert reg.data == data

        qubits = [Qubit(3), Qubit("pineapple")]
        reg.add(qubits)
        assert reg.data == data + qubits

    def test_add_duplicate_data(self):
        """Test adding qubits to the register using existing labels."""
        data = [Qubit("banana"), Qubit("mango")]
        reg = QuantumRegister(data)
        assert reg.data == data

        with pytest.raises(ValueError, match="already in register"):
            reg.add(data[0])

    def test_freeze(self):
        """Test freezing the register."""
        reg = QuantumRegister(data=[Qubit(0), Qubit(1)])
        reg.freeze()

        assert reg.frozen

        with pytest.raises(
            RegisterError, match="Register is frozen and no more data can be added."
        ):
            reg.add("abc")


class TestClassicalRegister:
    """Unit tests for the classical register class."""

    def test_empty_register(self):
        """Test initializing an empty data register."""
        reg = ClassicalRegister()

        assert reg.data == []
        assert len(reg) == 0

    def test_register_with_data(self):
        """Test initializing an empty data register."""
        data = [Bit(1), Bit(2), Bit(3), Bit(42)]
        reg = ClassicalRegister(data)

        assert reg.data == data
        assert len(reg) == 4

    def test_register_iteration(self):
        """Test iterating over a register."""
        data = [Bit(1), Bit(21), Bit(32)]
        reg = ClassicalRegister(data=data)

        for i, d in enumerate(reg):
            assert d == data[i]

    def test_index(self):
        """Test getting the index of an item in a register."""
        data = [Bit(1), Bit(13), Bit(42), Bit(54)]
        reg = ClassicalRegister(data=data)

        assert reg.index(data[1]) == data.index(data[1])

    def test_register_indexing(self):
        """Test getting an item at a specific index in a register."""
        data = [Bit(1), Bit(13), Bit(42), Bit(54)]
        reg = ClassicalRegister(data=data)

        # don't use enumerate so as to _only_ test indexing (and not iteration)
        for i in range(len(reg)):
            assert reg[i] == data[i]

    def test_repr(self):
        """Test the representation of a register."""
        data = [Bit(1), Bit(13), Bit(42), Bit(54)]
        reg = ClassicalRegister(data)

        assert reg.__repr__() == f"<ClassicalRegister, data={data}>"

    def test_str(self):
        """Test the string of a register."""
        data = [Bit(1), Bit(13), Bit(42), Bit(54)]
        reg = ClassicalRegister(data)

        assert str(reg) == f"ClassicalRegister({data})"

    def test_add(self):
        """Test adding bits to the register."""
        data = [Bit(1), Bit(0)]
        reg = ClassicalRegister(data)
        assert reg.data == data

        bits = [Bit(42), Bit(24)]
        reg.add(bits)
        assert reg.data == data + bits

    def test_add_duplicate_data(self):
        """Test adding bits to the register using existing labels."""
        data = [Bit("banana"), Bit("mango")]
        reg = ClassicalRegister(data)
        assert reg.data == data

        with pytest.raises(ValueError, match="already in register"):
            reg.add(data[0])

    def test_freeze(self):
        """Test freezing the register."""
        data = [Bit(1), Bit(0)]
        reg = ClassicalRegister(data)
        reg.freeze()

        assert reg.frozen

        with pytest.raises(
            RegisterError, match="Register is frozen and no more data can be added."
        ):
            reg.add(Bit("123"))


class TestSelfIncrementingRegister:
    """Unit tests for the ``SelfIncrementingRegister`` class."""

    def test_empty_register(self):
        """Test initializing an empty register."""
        reg = SelfIncrementingRegister()

        assert reg.data == []
        assert len(reg) == 0

    def test_register_with_data(self):
        """Test initializing an empty data register."""
        data = [Variable("a"), Variable("b"), Variable("c"), Variable("d")]
        reg = SelfIncrementingRegister(data)

        assert reg.data == data
        assert len(reg) == 4

    def test_register_iteration(self):
        """Test iterating over a register."""
        data = [Variable("a"), Variable("b"), Variable("c"), Variable("d")]
        reg = SelfIncrementingRegister(data)

        for i, d in enumerate(reg):
            assert d == data[i]

    def test_index(self):
        """Test getting the index of an item in a register."""
        data = [Variable("a"), Variable("b"), Variable("c"), Variable("d")]
        reg = SelfIncrementingRegister(data)

        assert reg.index(data[2]) == data.index(data[2])

    def test_register_indexing(self):
        """Test getting an item at a specific index in a register."""
        data = [Variable("a"), Variable("b"), Variable("c"), Variable("d")]
        reg = SelfIncrementingRegister(data)

        # don't use enumerate so as to _only_ test indexing (and not iteration)
        for i in range(len(reg)):
            assert reg[i] == data[i]

    def test_indexing_outside_scope(self):
        """Test getting an item at an index outside the scope of the register."""
        data = [Variable("a"), Variable("b")]
        reg = SelfIncrementingRegister(data)

        assert reg[4] == Variable("4")
        assert reg.data == data + [Variable("2"), Variable("3"), Variable("4")]

    def test_repr(self):
        """Test the representation of a register."""
        data = [Variable("a"), Variable("b"), Variable("c"), Variable("d")]
        reg = SelfIncrementingRegister(data)

        assert reg.__repr__() == f"<SelfIncrementingRegister, data={data}>"

    def test_str(self):
        """Test the string of a register."""
        data = [Variable("a"), Variable("b"), Variable("c"), Variable("d")]
        reg = SelfIncrementingRegister(data)

        assert str(reg) == f"SelfIncrementingRegister({data})"

    def test_add(self):
        """Test adding variables to the register."""
        data = [Variable("a"), Variable("b")]
        reg = SelfIncrementingRegister(data)
        assert reg.data == data

        vars = [Variable("c"), Variable("d")]
        reg.add(vars)
        assert reg.data == data + vars

    def test_add_duplicate_data(self):
        """Test adding bits to the register using existing labels."""
        data = [Variable("banana"), Variable("mango")]
        reg = SelfIncrementingRegister(data)
        assert reg.data == data

        # check that new variables using the same same are equal
        assert reg.data == [Variable("banana"), Variable("mango")]

        with pytest.raises(ValueError, match="already in register"):
            reg.add(Variable("mango"))

    def test_freeze(self):
        """Test freezing the register."""
        data = [Variable("a"), Variable("b")]
        reg = SelfIncrementingRegister(data)
        reg.freeze()

        assert reg.frozen

        with pytest.raises(
            RegisterError, match="Register is frozen and no more data can be added."
        ):
            reg.add(Variable("abc"))

    def test_selfincrementing_when_frozen(self):
        """Test freezing the register and then attempting to access outside of scope."""
        data = [Variable("a"), Variable("b")]
        reg = SelfIncrementingRegister(data)
        reg.freeze()

        assert reg.frozen

        with pytest.raises(IndexError):
            reg[15]
