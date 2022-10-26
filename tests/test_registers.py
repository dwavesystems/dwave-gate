# Confidential & Proprietary Information: D-Wave Systems Inc.
import pytest

from dwave.gate.registers import (
    ClassicalRegister,
    QuantumRegister,
    RegisterError,
    SelfIncrementingRegister,
    Variable,
)


class TestQuantumRegister:
    """Unit tests for the quantum register class."""

    def test_empty_register(self):
        """Test initializing an empty data register."""
        reg = QuantumRegister(label="reg")

        assert reg.label == "reg"
        assert reg.data == []
        assert len(reg) == 0

    def test_register_with_data(self):
        """Test initializing an empty data register."""
        reg = QuantumRegister(label="reg", data=[1, 2, 3, 42])

        assert reg.label == "reg"
        assert reg.data == [1, 2, 3, 42]
        assert len(reg) == 4

    def test_register_iteration(self):
        """Test iterating over a register."""
        data = [1, 21, 32]
        reg = QuantumRegister(label="reg", data=data)

        for i, d in enumerate(reg):
            assert d == data[i]

    def test_index(self):
        """Test getting the index of an item in a register."""
        data = [1, 13, 42, 54]
        reg = QuantumRegister(label="reg", data=data)

        assert reg.index(13) == data.index(13)

    def test_register_indexing(self):
        """Test getting an item at a specific index in a register."""
        data = [1, 13, 42, 54]
        reg = QuantumRegister(label="reg", data=data)

        # don't use enumerate so as to _only_ test indexing (and not iteration)
        for i in range(len(reg)):
            assert reg[i] == data[i]

    def test_register_slicing(self):
        """Test slicing a register."""
        reg = QuantumRegister(label="reg", data=[1, 13, 42, 54])

        assert reg[:] == [1, 13, 42, 54]
        assert reg[1:] == [13, 42, 54]
        assert reg[2:100] == [42, 54]
        assert reg[-1] == 54

    def test_register_indexing_out_of_range(self):
        """Test getting a item at a index out-of-range in a register."""
        reg = QuantumRegister(label="reg", data=[1, 13, 42, 54])

        assert reg[42] is None
        assert reg[4] is None

    def test_contains(self):
        """Test checking if an item is in a register."""
        reg = QuantumRegister(label="reg", data=[1, 13, 42, 54])

        assert 13 in reg
        assert 657 not in reg
        assert "qbut" not in reg

    def test_repr(self):
        """Test the representation of a register."""
        reg = QuantumRegister(label="reg", data=[1, 13, 42, 54])

        assert reg.__repr__() == "<QuantumRegister, data=[1, 13, 42, 54]>"

    def test_str(self):
        """Test the string of a register."""
        reg = QuantumRegister(label="reg", data=[1, 13, 42, 54])

        assert str(reg) == "[1, 13, 42, 54]"

    @pytest.mark.parametrize(
        "qubits, expected",
        [
            ("q0", ["q0"]),
            (("qb0, qb1, qb2"), ["qb0, qb1, qb2"]),
            (["banana", "mango"], ["banana", "mango"]),
        ],
    )
    def test_add(self, qubits, expected):
        """Test adding qubits to the register."""
        reg = QuantumRegister(label="reg", data=[0, 1])
        assert reg.data == [0, 1]

        reg.add(qubits)
        assert reg.data == [0, 1] + expected

    @pytest.mark.parametrize("qubits", ["mango", ["banana", "mango"], ["coconut", "coconut"]])
    def test_add_duplicate_data(self, qubits):
        """Test adding qubits to the register using existing labels."""
        reg = QuantumRegister(label="reg", data=["banana", "mango"])
        assert reg.data == ["banana", "mango"]

        with pytest.raises(ValueError, match="already in use"):
            reg.add(qubits)

    def test_freeze(self):
        """Test freezing the register."""
        reg = QuantumRegister(label="reg", data=[0, 1])
        reg.freeze()

        assert reg.frozen

        with pytest.raises(
            RegisterError, match="Register is frozen and no more data can be addded."
        ):
            reg.add("abc")


class TestClassicalRegister:
    """Unit tests for the classical register class."""

    def test_empty_register(self):
        """Test initializing an empty data register."""
        reg = ClassicalRegister(label="reg")

        assert reg.label == "reg"
        assert reg.data == []
        assert len(reg) == 0

    def test_register_with_data(self):
        """Test initializing an empty data register."""
        reg = ClassicalRegister(label="reg", data=[1, 2, 3, 42])

        assert reg.label == "reg"
        assert reg.data == [1, 2, 3, 42]
        assert len(reg) == 4

    def test_register_iteration(self):
        """Test iterating over a register."""
        data = [1, 21, 32]
        reg = ClassicalRegister(label="reg", data=data)

        for i, d in enumerate(reg):
            assert d == data[i]

    def test_index(self):
        """Test getting the index of an item in a register."""
        data = [1, 13, 42, 54]
        reg = ClassicalRegister(label="reg", data=data)

        assert reg.index(13) == data.index(13)

    def test_register_indexing(self):
        """Test getting an item at a specific index in a register."""
        data = [1, 13, 42, 54]
        reg = ClassicalRegister(label="reg", data=data)

        # don't use enumerate so as to _only_ test indexing (and not iteration)
        for i in range(len(reg)):
            assert reg[i] == data[i]

    def test_repr(self):
        """Test the representation of a register."""
        reg = ClassicalRegister(label="reg", data=[1, 13, 42, 54])

        assert reg.__repr__() == "<ClassicalRegister, data=[1, 13, 42, 54]>"

    def test_str(self):
        """Test the string of a register."""
        reg = ClassicalRegister(label="reg", data=[1, 13, 42, 54])

        assert str(reg) == "[1, 13, 42, 54]"

    @pytest.mark.parametrize(
        "bits, expected",
        [
            ("c0", ["c0"]),
            (("cb0, cb1, cb2"), ["cb0, cb1, cb2"]),
            (["banana", "mango"], ["banana", "mango"]),
        ],
    )
    def test_add(self, bits, expected):
        """Test adding bits to the register."""
        reg = ClassicalRegister(label="reg", data=[0, 1])
        assert reg.data == [0, 1]

        reg.add(bits)
        assert reg.data == [0, 1] + expected

    @pytest.mark.parametrize("bits", ["mango", ["banana", "mango"], ["coconut", "coconut"]])
    def test_add_duplicate_data(self, bits):
        """Test adding bits to the register using existing labels."""
        reg = ClassicalRegister(label="reg", data=["banana", "mango"])
        assert reg.data == ["banana", "mango"]

        with pytest.raises(ValueError, match="already in use"):
            reg.add(bits)

    def test_freeze(self):
        """Test freezing the register."""
        reg = ClassicalRegister(label="reg", data=[0, 1])
        reg.freeze()

        assert reg.frozen

        with pytest.raises(
            RegisterError, match="Register is frozen and no more data can be addded."
        ):
            reg.add("abc")


class TestSelfIncrementingRegister:
    """Unit tests for the ``SelfIncrementingRegister`` class."""

    def test_empty_register(self):
        """Test initializing an empty register."""
        reg = SelfIncrementingRegister(label="reg")

        assert reg.label == "reg"
        assert reg.data == []
        assert len(reg) == 0

    def test_register_with_data(self):
        """Test initializing an empty data register."""
        reg = SelfIncrementingRegister(label="reg", data=[1, 2, 3, 42])

        assert reg.label == "reg"
        assert reg.data == [1, 2, 3, 42]
        assert len(reg) == 4

    def test_register_iteration(self):
        """Test iterating over a register."""
        data = [1, 21, 32]
        reg = SelfIncrementingRegister(label="reg", data=data)

        for i, d in enumerate(reg):
            assert d == data[i]

    def test_index(self):
        """Test getting the index of an item in a register."""
        data = [1, 13, 42, 54]
        reg = SelfIncrementingRegister(label="reg", data=data)

        assert reg.index(13) == data.index(13)

    def test_register_indexing(self):
        """Test getting an item at a specific index in a register."""
        data = [1, 13, 42, 54]
        reg = SelfIncrementingRegister(label="reg", data=data)

        # don't use enumerate so as to _only_ test indexing (and not iteration)
        for i in range(len(reg)):
            assert reg[i] == data[i]

    def test_indexing_outside_scope(self):
        """Test getting an item at an index outside the scope of the register."""
        data = [1, 13]
        reg = SelfIncrementingRegister(label="reg", data=data)

        assert reg[4] == Variable("4")
        assert reg.data == data + [Variable("2"), Variable("3"), Variable("4")]

    def test_repr(self):
        """Test the representation of a register."""
        reg = SelfIncrementingRegister(label="reg", data=[1, 13, 42, 54])

        assert reg.__repr__() == "<SelfIncrementingRegister, data=[1, 13, 42, 54]>"

    def test_str(self):
        """Test the string of a register."""
        reg = SelfIncrementingRegister(label="reg", data=[1, 13, 42, 54])

        assert str(reg) == "[1, 13, 42, 54]"

    @pytest.mark.parametrize(
        "bits, expected",
        [
            ("c0", ["c0"]),
            (("cb0, cb1, cb2"), ["cb0, cb1, cb2"]),
            (["banana", "mango"], ["banana", "mango"]),
        ],
    )
    def test_add(self, bits, expected):
        """Test adding bits to the register."""
        reg = SelfIncrementingRegister(label="reg", data=[0, 1])
        assert reg.data == [0, 1]

        reg.add(bits)
        assert reg.data == [0, 1] + expected

    @pytest.mark.parametrize("bits", ["mango", ["banana", "mango"], ["coconut", "coconut"]])
    def test_add_duplicate_data(self, bits):
        """Test adding bits to the register using existing labels."""
        reg = SelfIncrementingRegister(label="reg", data=["banana", "mango"])
        assert reg.data == ["banana", "mango"]

        with pytest.raises(ValueError, match="already in use"):
            reg.add(bits)

    def test_freeze(self):
        """Test freezing the register."""
        reg = SelfIncrementingRegister(label="reg", data=[0, 1])
        reg.freeze()

        assert reg.frozen

        with pytest.raises(
            RegisterError, match="Register is frozen and no more data can be addded."
        ):
            reg.add("abc")

    def test_selfincrementing_when_frozen(self):
        """Test freezing the register and then attempting to access outside of scope."""
        reg = SelfIncrementingRegister(label="reg", data=[0, 1])
        reg.freeze()

        assert reg.frozen

        with pytest.raises(IndexError):
            reg[15]


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
