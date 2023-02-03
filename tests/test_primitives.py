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


class TestQubit:
    """Unit tests for the ``Qubit`` class."""

    def test_initialize(self):
        """Test initializing a qubit."""
        label = "ananas"
        qubit = Qubit(label)

        assert qubit.label == label

    def test_set_label(self):
        """Test set qubit label."""
        qubit = Qubit("ananas")
        assert qubit.label == "ananas"

        qubit.label = "banana"
        assert qubit.label == "banana"

    def test_equality(self):
        "Test equality of two qubits with the same label."
        label = "ananas"
        bit_0 = Qubit(label)
        bit_1 = Qubit(label)

        assert bit_0 != bit_1
        assert bit_0 is not bit_1

    def test_repr(self):
        "Test the representation of a qubit."
        qubit = Qubit("ananas")
        assert qubit.__repr__() == f"<qubit: 'ananas', id: {qubit.id}>"


class TestBit:
    """Unit tests for the ``Bit`` class."""

    def test_initialize(self):
        """Test initializing a bit."""
        label = "ananas"
        bit = Bit(label)

        assert bit.label == label

    def test_set_label(self):
        """Test set bit label."""
        bit = Bit("ananas")
        assert bit.label == "ananas"

        bit.label = "banana"
        assert bit.label == "banana"

    def test_equality(self):
        "Test equality of two bits with the same label."
        label = "ananas"
        bit_0 = Bit(label)
        bit_1 = Bit(label)

        assert bit_0 != bit_1
        assert bit_0 is not bit_1

    def test_repr(self):
        "Test the representation of a bit."
        bit = Bit("ananas")
        assert bit.__repr__() == f"<bit: 'ananas', id: {bit.id}>"

    @pytest.mark.parametrize("value", [1, 0, "1", True, 42, (1, 2, 3)])
    def test_set_value(self, value):
        """Test setting a value for the bit."""
        bit = Bit("banana")
        bit.set(value)

        assert bit == int(bool(value))
        assert bit.__repr__() == f"<bit: 'banana', id: {bit.id}, value: {int(bool(value))}>"

        bit.reset()

        assert bit != 3.14

    def test_force_set_value(self):
        """Test forcing a value for the bit."""
        bit = Bit("banana")
        bit.set(1)
        assert bit.value == 1

        with pytest.raises(ValueError, match="Value already set"):
            bit.set(0)

        bit.set(0, force=True)
        assert bit.value == 0

    @pytest.mark.parametrize("value", [1, 0, "1", True])
    def test_bool(self, value):
        """Test the boolean represention of the bit."""
        bit = Bit("coconut")
        bit.set(value)

        assert bool(bit) == bool(value)


class TestVariable:
    """Unit tests for the ``Variable`` class."""

    def test_initialize(self):
        """Test initializing a variable."""
        name = "ananas"
        var = Variable(name)

        assert var.name == name

    def test_hash(self):
        "Test that two equal variables have the same hash."
        name = "ananas"
        var_0 = Variable(name)
        var_1 = Variable(name)

        assert var_0 == var_1
        assert hash(var_0) == hash(var_1)

    def test_equality(self):
        "Test equality of two variables with the same name."
        name = "ananas"
        var_0 = Variable(name)
        var_1 = Variable(name)

        assert var_0 == var_1
        assert var_0 is not var_1

        assert var_0 != name

    def test_repr(self):
        "Test the representation of a variable."
        var = Variable("ananas")
        assert var.__repr__() == "{ananas}"

    def test_set_value(self):
        """Test setting a value for the variable."""
        var = Variable("banana")
        assert var.__repr__() == "{banana}"

        var.set(3.14)

        assert var == 3.14
        assert var.__repr__() == "3.14"

        var.reset()

        assert var != 3.14
        assert var.__repr__() == "{banana}"
