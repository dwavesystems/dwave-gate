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

import numpy as np
import pytest

import dwave.gate.operations as ops
from dwave.gate import CircuitContext
from dwave.gate.circuit import Circuit, CircuitError, ParametricCircuit, ParametricCircuitContext
from dwave.gate.primitives import Bit, Qubit, Variable
from dwave.gate.registers.registers import ClassicalRegister, QuantumRegister


class TestCircuitError:
    def test_circuit_error(self):
        with pytest.raises(CircuitError, match="testing error"):
            raise CircuitError("testing error")


class TestCircuit:
    """Unit tests for the Circuit class."""

    @pytest.mark.parametrize("qubits, bits", [(0, 0), (4, 0), (0, 6), (2, 2)])
    def test_qubits_and_bits(self, qubits, bits):
        """Test circuit with bits and qubits."""
        qubits, bits = 3, 2
        circuit = Circuit(qubits, bits)

        assert circuit.num_qubits == qubits
        assert circuit.num_bits == bits

        # check (qu)bits and their labels in the circuit
        for qb, b, label in zip(circuit.qubits, circuit.bits, [str(i) for i in range(qubits)]):
            assert qb.label == b.label == label

        # check registers in the circuit
        assert isinstance(circuit.qregisters["qreg0"], QuantumRegister)
        assert isinstance(circuit.cregisters["creg0"], ClassicalRegister)

    def test_qubits(self):
        """Test circuit with qubits."""
        circuit = Circuit(num_qubits=4)

        assert circuit.num_qubits == 4
        assert circuit.num_bits == 0

        assert [qb.label for qb in circuit.qubits] == ["0", "1", "2", "3"]
        assert isinstance(circuit.qregisters["qreg0"], QuantumRegister)

    def test_bits(self):
        """Test circuit with bits."""
        circuit = Circuit(num_bits=3)

        assert circuit.num_qubits == 0
        assert circuit.num_bits == 3

        assert [b.label for b in circuit.bits] == ["0", "1", "2"]
        assert isinstance(circuit.cregisters["creg0"], ClassicalRegister)

    @pytest.mark.parametrize("force", [False, True])
    def test_set_state(self, force):
        """Test the ``Circuit.set_state`` method."""
        circuit = Circuit(2)
        state = np.array([1, 2, 3, 4]) / np.sqrt(30)
        circuit.set_state(state, force=force, normalize=False)

        assert circuit._density_matrix is None
        assert np.allclose(circuit.state, state)

    @pytest.mark.parametrize("force", [False, True])
    def test_set_state_normalize(self, force):
        """Test the ``Circuit.set_state`` method with an unnormalized state."""
        circuit = Circuit(2)
        state = np.array([1, 2, 3, 4])
        circuit.set_state(state, force=force, normalize=True)

        assert circuit._density_matrix is None
        assert np.allclose(circuit.state, state / np.sqrt(30))

    @pytest.mark.parametrize("force", [False, True])
    def test_set_dm(self, force):
        """Test ``Circuit.set_state`` with a density matrix."""
        circuit = Circuit(2)
        state = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]) / 30
        circuit.set_state(state, force=force, normalize=False)

        with pytest.raises(CircuitError, match="State is mixed."):
            circuit.state
        assert np.allclose(circuit.density_matrix, state)

    @pytest.mark.parametrize("force", [False, True])
    def test_set_dm_normalize(self, force):
        """Test ``Circuit.set_state`` with an unnormalized density matrix."""
        circuit = Circuit(2)
        state = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]])
        circuit.set_state(state, force=force, normalize=True)

        with pytest.raises(CircuitError, match="State is mixed."):
            circuit.state
        assert np.allclose(circuit.density_matrix, state / 30)

    @pytest.mark.parametrize("force", [False, True])
    def test_set_state_normalize_error(self, force):
        """Test that the correct error is raised when setting an unnormalized state."""
        circuit = Circuit(2)
        state = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="State is not normalized."):
            circuit.set_state(state, force=force, normalize=False)

    @pytest.mark.parametrize("force", [False, True])
    def test_set_state_shape_error(self, force):
        """Test that the correct error is raised when setting a state with incorrect shape."""
        circuit = Circuit(2)
        state = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # 3-qubit state
        with pytest.raises(ValueError, match="State has incorrect shape."):
            circuit.set_state(state, force=force, normalize=False)

    def test_force_set_state(self):
        """Test the ``Circuit.set_state`` method twice."""
        circuit = Circuit(2)
        state_0 = np.array([1, 2, 3, 4]) / np.sqrt(30)
        state_1 = np.array([4, 3, 2, 1]) / np.sqrt(30)

        circuit.set_state(state_0, force=False)

        with pytest.raises(CircuitError, match="State already set."):
            circuit.set_state(state_1, force=False)

        circuit.set_state(state_1, force=True)

        assert circuit._density_matrix is None
        assert np.allclose(circuit.state, state_1)

    @pytest.mark.parametrize("force", [False, True])
    def test_get_dm_from_sv(self, force):
        """Test getting the density matrix from a state vector."""
        circuit = Circuit(2)
        state = np.array([1, 2, 3, 4]) / np.sqrt(30)
        circuit.set_state(state, force=force)

        expected_dm = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]) / 30
        assert np.allclose(circuit.density_matrix, expected_dm)

    def test_get_qubit(self):
        """Test finding and getting a qubit using the label."""
        circuit = Circuit()
        circuit.add_qregister(2, label="fruits")

        apple_qubit = Qubit("apple")
        circuit.add_qubit(apple_qubit, "fruits")

        qubit = circuit.get_qubit("apple")
        assert qubit == apple_qubit

    def test_get_qubit_specific_register(self):
        """Test finding and getting a qubit in a specified register."""
        circuit = Circuit()
        circuit.add_qregister(2, label="fruits")
        circuit.add_qregister(3, label="vegetables")

        fruit_qubit = Qubit("tomato")
        circuit.add_qubit(fruit_qubit, "fruits")
        vegetable_qubit = Qubit("tomato")
        circuit.add_qubit(vegetable_qubit, "vegetables")

        qubit = circuit.get_qubit("tomato", qreg_label="vegetables")
        assert qubit == vegetable_qubit

        qubit = circuit.get_qubit("tomato", qreg_label="fruits")
        assert qubit == fruit_qubit

    def test_get_qubit_not_found(self):
        """Test that the correct error is raised when getting a
        non-existing qubit using the label."""
        circuit = Circuit()
        circuit.add_qregister(2, label="fruits")

        with pytest.raises(ValueError, match="Qubit 'apple' not found."):
            qubit = circuit.get_qubit("apple")

    def test_get_multiple_qubits(self):
        """Test finding and getting a qubit when there are multiple qubits
        using the same label."""
        circuit = Circuit()
        circuit.add_qregister(2, label="fruits")

        apple_qubit_0 = Qubit("apple")
        apple_qubit_1 = Qubit("apple")
        circuit.add_qubit(apple_qubit_0, "fruits")
        circuit.add_qubit(apple_qubit_1, "fruits")

        qubit = circuit.get_qubit("apple")
        assert qubit == apple_qubit_0

    def test_get_multiple_qubits_return_all(self):
        """Test finding and getting multiple qubits using the same label."""
        circuit = Circuit()
        circuit.add_qregister(2, label="fruits")

        apple_qubit_0 = Qubit("apple")
        apple_qubit_1 = Qubit("apple")
        circuit.add_qubit(apple_qubit_0, "fruits")
        circuit.add_qubit(apple_qubit_1, "fruits")

        qubits = circuit.get_qubit("apple", return_all=True)
        assert qubits == [apple_qubit_0, apple_qubit_1]

    def test_find_qubit(self):
        """Test finding a qubit in a circuit."""
        circuit = Circuit()
        circuit.add_qregister(label="fruits")
        circuit.add_qregister(label="gemstones")

        kumquat_qubit = Qubit("kumquat")
        circuit.add_qubit(Qubit("dummy_qubit"), "fruits")
        circuit.add_qubit(kumquat_qubit, "fruits")
        emerald_qubit = Qubit("emerald")
        circuit.add_qubit(emerald_qubit, "gemstones")

        qreg_label, idx = circuit.find_qubit(kumquat_qubit, qreg_label=True)
        assert qreg_label == "fruits"
        assert idx == 1  # after 'dummy_qubit'

        qreg_label, idx = circuit.find_qubit(emerald_qubit, qreg_label=True)
        assert qreg_label == "gemstones"
        assert idx == 0  # only qubit in register

    def test_find_qubit_no_label(self):
        """Test finding a qubit in a circuit returning the register index."""
        circuit = Circuit()
        circuit.add_qregister()  # first qreg
        circuit.add_qregister(label="fruits")  # second qreg

        qubit = Qubit("apple")
        circuit.add_qubit(qubit, "fruits")

        qreg_idx, idx = circuit.find_qubit(qubit, qreg_label=False)
        assert qreg_idx == 1
        assert idx == 0

    def test_find_qubit_not_found(self):
        """Test that the correct exception is raised when qubit is not found."""
        circuit = Circuit()
        circuit.add_qregister()

        with pytest.raises(ValueError, match="not found in any register"):
            circuit.find_qubit(Qubit("apple"), qreg_label=False)

    def test_get_bit(self):
        """Test finding and getting a bit using the label."""
        circuit = Circuit()
        circuit.add_cregister(2, label="fruits")

        apple_bit = Bit("apple")
        circuit.add_bit(apple_bit, "fruits")

        bit = circuit.get_bit("apple")
        assert bit == apple_bit

    def test_get_bit_specific_register(self):
        """Test finding and getting a bit in a specified register."""
        circuit = Circuit()
        circuit.add_cregister(2, label="fruits")
        circuit.add_cregister(3, label="vegetables")

        fruit_bit = Bit("tomato")
        circuit.add_bit(fruit_bit, "fruits")
        vegetable_bit = Bit("tomato")
        circuit.add_bit(vegetable_bit, "vegetables")

        bit = circuit.get_bit("tomato", creg_label="vegetables")
        assert bit == vegetable_bit

        bit = circuit.get_bit("tomato", creg_label="fruits")
        assert bit == fruit_bit

    def test_get_bit_not_found(self):
        """Test that the correct error is raised when getting a
        non-existing bit using the label."""
        circuit = Circuit()
        circuit.add_cregister(2, label="fruits")

        with pytest.raises(ValueError, match="Bit 'apple' not found."):
            bit = circuit.get_bit("apple")

    def test_get_multiple_bits(self):
        """Test finding and getting a bit when there are multiple bits
        using the same label."""
        circuit = Circuit()
        circuit.add_cregister(2, label="fruits")

        apple_bit_0 = Bit("apple")
        apple_bit_1 = Bit("apple")
        circuit.add_bit(apple_bit_0, "fruits")
        circuit.add_bit(apple_bit_1, "fruits")

        bit = circuit.get_bit("apple")
        assert bit == apple_bit_0

    def test_get_multiple_bits_return_all(self):
        """Test finding and getting multiple bits using the same label."""
        circuit = Circuit()
        circuit.add_cregister(2, label="fruits")

        apple_bit_0 = Bit("apple")
        apple_bit_1 = Bit("apple")
        circuit.add_bit(apple_bit_0, "fruits")
        circuit.add_bit(apple_bit_1, "fruits")

        bits = circuit.get_bit("apple", return_all=True)
        assert bits == [apple_bit_0, apple_bit_1]

    def test_find_bit_not_found(self):
        """Test that the correct exception is raised when bit is not found."""
        circuit = Circuit()
        circuit.add_cregister()

        with pytest.raises(ValueError, match="not found in any register"):
            circuit.find_bit(Bit("apple"), creg_label=False)

    def test_find_bit_no_label(self):
        """Test finding a qubit in a circuit returning the register index."""
        circuit = Circuit()
        circuit.add_cregister()  # first qreg
        circuit.add_cregister(label="fruits")  # second qreg

        bit = Bit("apple")
        circuit.add_bit(bit, "fruits")

        creg_idx, idx = circuit.find_bit(bit, creg_label=False)
        assert creg_idx == 1
        assert idx == 0

    def test_find_bit(self):
        """Test finding a bit in a circuit."""
        circuit = Circuit()
        circuit.add_cregister(label="fruits")
        circuit.add_cregister(label="gemstones")

        kumquat_bit = Bit("kumquat")
        circuit.add_bit(Bit("dummy_bit"), "fruits")
        circuit.add_bit(kumquat_bit, "fruits")
        emerald_bit = Bit("emerald")
        circuit.add_bit(emerald_bit, "gemstones")

        creg_label, idx = circuit.find_bit(kumquat_bit, creg_label=True)
        assert creg_label == "fruits"
        assert idx == 1  # after 'dummy_bit'

        creg_label, idx = circuit.find_bit(emerald_bit, creg_label=True)
        assert creg_label == "gemstones"
        assert idx == 0  # only qubit in register

    def test_representation(self):
        """Test represenation of a circuit."""
        circuit = Circuit(2)
        assert circuit.__repr__() == f"<Circuit: qubits=2, bits=0, ops=0>"

    def test_appending_operations(self):
        """Test adding operations to a circuit."""
        circuit = Circuit(2)
        operations = [
            ops.X(circuit.qubits[0]),
            ops.Y(circuit.qubits[1]),
            ops.CX(circuit.qubits[0], circuit.qubits[1]),
        ]

        circuit.extend(operations)

        assert circuit.circuit == operations

    def test_appending_operations_locked(self, two_qubit_circuit):
        """Test adding operations to a locked circuit."""
        op = ops.X(two_qubit_circuit.qubits[0])

        two_qubit_circuit.lock()
        with pytest.raises(CircuitError, match="Circuit is locked"):
            two_qubit_circuit.append(op)

        two_qubit_circuit.unlock()
        two_qubit_circuit.append(op)
        assert op in two_qubit_circuit.circuit

    def test_appending_operations_invalid_qubits(self):
        """Test adding operations on invalid qubits to a circuit."""
        circuit = Circuit(2)

        with pytest.raises(ValueError, match="not in circuit."):
            circuit.append(ops.X(Qubit("coconut")))

    def test_remove(self):
        """Test removing an operation from a circuit."""
        circuit = Circuit(2)

        # add two operations to make sure only a single op can be removed
        op = ops.X(circuit.qubits[0])
        op_extra = ops.CNOT(circuit.qubits[0], circuit.qubits[1])

        circuit.append(op)
        circuit.append(op_extra)
        assert circuit.circuit == [op, op_extra]

        circuit.remove(op)
        assert circuit.circuit == [op_extra]

    def test_invalid_remove(self):
        """Test removing an operation which is not part of the circuit."""
        circuit = Circuit(2)
        op = ops.X(circuit.qubits[0])

        circuit.append(op)
        assert circuit.circuit == [op]

        with pytest.raises(ValueError, match="not in circuit"):
            circuit.remove(ops.Y(circuit.qubits[0]))

        assert circuit.circuit == [op]

    def test_context(self, two_qubit_circuit):
        """Test that a context is returned correctly."""
        context = two_qubit_circuit.context

        assert isinstance(context, CircuitContext)

        # assert that the same context is always returned
        assert context is two_qubit_circuit.context

    @pytest.mark.parametrize("keep_registers", [True, False])
    def test_reset(self, two_qubit_circuit, keep_registers):
        """Test resetting a circuit."""
        assert two_qubit_circuit.circuit
        context = two_qubit_circuit.context

        assert two_qubit_circuit.qregisters
        assert two_qubit_circuit.cregisters == dict()  # contains no bits

        two_qubit_circuit.reset(keep_registers=keep_registers)

        assert two_qubit_circuit.circuit == []
        assert context is not two_qubit_circuit.context

        if keep_registers:
            assert two_qubit_circuit.qregisters
            assert two_qubit_circuit.cregisters == dict()  # contains no bits
        else:
            assert two_qubit_circuit.qregisters == dict()
            assert two_qubit_circuit.cregisters == dict()

    def test_reset_bits(self):
        """Test resetting a circuit with bits that contain measurement values."""
        circuit = Circuit(2, 2)
        with circuit.context as (q, c):
            ops.X(q[0])
            ops.Measurement(q[0]) | c[0]

        # must simulate circuit to populate classical register
        from dwave.gate.simulator.simulator import simulate

        assert c[0].value is None
        simulate(circuit)
        assert c[0].value is not None
        circuit.reset()
        assert c[0].value is None

    def test_add_qubit(self, two_qubit_circuit):
        """Test adding qubits to a circuit."""
        assert two_qubit_circuit.num_qubits == 2
        assert [qb.label for qb in two_qubit_circuit.qubits] == ["0", "1"]

        two_qubit_circuit.add_qubit(Qubit("pineapple"))
        two_qubit_circuit.add_qubit()

        assert two_qubit_circuit.num_qubits == 4
        assert [qb.label for qb in two_qubit_circuit.qubits] == ["0", "1", "pineapple", "3"]

    @pytest.mark.parametrize("not_a_qubit", ["just_a_label", 3, Bit("c0")])
    def test_add_qubit_typerror(self, two_qubit_circuit, not_a_qubit):
        """Test that the correct error is raised when adding the wrong type."""
        with pytest.raises(TypeError, match="Can only add qubits to circuit."):
            two_qubit_circuit.add_qubit(not_a_qubit)

    def test_add_qubit_to_empty_circuit(self, empty_circuit):
        """Test adding qubits to an empty circuit."""
        assert empty_circuit.num_qubits == 0
        assert len(empty_circuit.qubits) == 0

        empty_circuit.add_qubit(Qubit("pineapple"))
        empty_circuit.add_qubit()

        assert empty_circuit.num_qubits == 2
        assert [qb.label for qb in empty_circuit.qubits] == ["pineapple", "1"]

    def test_add_qubit_with_qreg_label(self, two_qubit_circuit):
        """Test adding qubits to a circuit specifying a qreg label."""
        assert two_qubit_circuit.num_qubits == 2
        assert [qb.label for qb in two_qubit_circuit.qubits] == ["0", "1"]

        # add a qubit to a non-existing qreg
        two_qubit_circuit.add_qubit(Qubit("pineapple"), qreg_label="coconut")
        # add a qubit to an already existing qreg (default name)
        two_qubit_circuit.add_qubit(Qubit("crabapple"), qreg_label="qreg0")

        assert two_qubit_circuit.num_qubits == 4
        # the order of qubits depend on the order of the qregs
        # 'qreg0' qubits first, and then 'coconut' qubits
        assert [qb.label for qb in two_qubit_circuit.qubits] == ["0", "1", "crabapple", "pineapple"]
        assert [qb.label for qb in two_qubit_circuit.qregisters["coconut"].data] == ["pineapple"]
        assert [qb.label for qb in two_qubit_circuit.qregisters["qreg0"].data] == [
            "0",
            "1",
            "crabapple",
        ]

    @pytest.mark.parametrize("qreg_label", [None, "qreg0", "coconut"])
    def test_add_qubit_with_existing_name(self, two_qubit_circuit, qreg_label):
        """Test adding a qubit to a circuit using an already existing label."""
        qubit = two_qubit_circuit.qregisters["qreg0"][0]
        assert qubit in two_qubit_circuit.qregisters["qreg0"].data

        with pytest.raises(ValueError, match="already in use"):
            two_qubit_circuit.add_qubit(qubit, qreg_label=qreg_label)

    def test_add_bit(self, two_bit_circuit):
        """Test adding bits to a circuit."""
        assert two_bit_circuit.num_bits == 2
        assert [b.label for b in two_bit_circuit.bits] == ["0", "1"]

        two_bit_circuit.add_bit(Bit("pineapple"))
        two_bit_circuit.add_bit()

        assert two_bit_circuit.num_bits == 4
        assert [b.label for b in two_bit_circuit.bits] == ["0", "1", "pineapple", "3"]

    @pytest.mark.parametrize("not_a_bit", ["just_a_label", 3, Qubit("q0")])
    def test_add_bit_typerror(self, two_bit_circuit, not_a_bit):
        """Test that the correct error is raised when adding the wrong type."""
        with pytest.raises(TypeError, match="Can only add bits to circuit."):
            two_bit_circuit.add_bit(not_a_bit)

    def test_add_bit_to_empty_circuit(self, empty_circuit):
        """Test adding bits to a circuit."""
        assert empty_circuit.num_bits == 0
        assert len(empty_circuit.bits) == 0

        empty_circuit.add_bit(Bit("pineapple"))
        empty_circuit.add_bit()

        assert empty_circuit.num_bits == 2
        assert [b.label for b in empty_circuit.bits] == ["pineapple", "1"]

    def test_add_bit_with_creg_label(self, two_bit_circuit):
        """Test adding bits to a circuit specifying a creg label."""
        assert two_bit_circuit.num_bits == 2
        assert [b.label for b in two_bit_circuit.bits] == ["0", "1"]

        # add a bit to a non-existing creg
        two_bit_circuit.add_bit(Bit("pineapple"), creg_label="coconut")
        # add a bit to an already existing creg (default name)
        two_bit_circuit.add_bit(Bit("crabapple"), creg_label="creg0")

        assert two_bit_circuit.num_bits == 4
        # the order of qubits depend on the order of the qregs
        # 'creg0' bits first, and then 'coconut' bits
        assert [b.label for b in two_bit_circuit.bits] == ["0", "1", "crabapple", "pineapple"]
        assert [b.label for b in two_bit_circuit.cregisters["coconut"].data] == ["pineapple"]
        assert [b.label for b in two_bit_circuit.cregisters["creg0"].data] == [
            "0",
            "1",
            "crabapple",
        ]

    @pytest.mark.parametrize("creg_label", [None, "creg0", "coconut"])
    def test_add_bit_with_existing_label(self, two_bit_circuit, creg_label):
        """Test adding a bit to a circuit using an already existing label."""
        bit = two_bit_circuit.cregisters["creg0"].data[0]
        assert bit in two_bit_circuit.cregisters["creg0"].data

        with pytest.raises(ValueError, match="already in use"):
            two_bit_circuit.add_bit(bit, creg_label=creg_label)

    @pytest.mark.parametrize("num_qubits", [0, 1, 5])
    def test_add_qregister(self, two_qubit_circuit, num_qubits):
        """Test adding a new qregister to the circuit."""
        assert len(two_qubit_circuit.qregisters) == 1

        two_qubit_circuit.add_qregister(num_qubits=num_qubits)
        assert len(two_qubit_circuit.qregisters) == 2
        assert len(two_qubit_circuit.qregisters["qreg1"]) == num_qubits

        two_qubit_circuit.add_qregister(num_qubits=num_qubits, label="coconut")
        assert len(two_qubit_circuit.qregisters) == 3
        assert len(two_qubit_circuit.qregisters["coconut"]) == num_qubits

    def test_add_qregister_with_existing_label(self, two_qubit_circuit):
        """Test adding a new qregister to the circuit using an already existing label."""
        with pytest.raises(ValueError, match="already present in the circuit"):
            two_qubit_circuit.add_qregister(label="qreg0")

    @pytest.mark.parametrize("num_bits", [0, 1, 5])
    def test_add_cregister(self, two_bit_circuit, num_bits):
        """Test adding a new cregister to the circuit."""
        assert len(two_bit_circuit.cregisters) == 1

        two_bit_circuit.add_cregister(num_bits=num_bits)
        assert len(two_bit_circuit.cregisters) == 2
        assert len(two_bit_circuit.cregisters["creg1"]) == num_bits

        two_bit_circuit.add_cregister(num_bits=num_bits, label="coconut")
        assert len(two_bit_circuit.cregisters) == 3
        assert len(two_bit_circuit.cregisters["coconut"]) == num_bits

    def test_add_cregister_with_existing_label(self, two_bit_circuit):
        """Test adding a new cregister to the circuit using an already existing label."""
        with pytest.raises(ValueError, match="already present in the circuit"):
            two_bit_circuit.add_cregister(label="creg0")

    def test_call_operation_instance(self):
        """Test calling an operation instance."""
        circuit = Circuit(3)
        with circuit.context as reg:
            ops.CNOT(reg.q[0], reg.q[1])(reg.q[1], reg.q[2])(reg.q[2], reg.q[0])

        q0, q1, q2 = circuit.qubits

        assert len(circuit.circuit) == 3
        assert circuit.circuit[0] == ops.CNOT(q0, q1)
        assert circuit.circuit[1] == ops.CNOT(q1, q2)
        assert circuit.circuit[2] == ops.CNOT(q2, q0)

    def test_call(self):
        """Test calling the circuit within a context to apply it to the active context."""
        circuit_1 = Circuit(2)
        operations = [
            ops.X(circuit_1.qubits[0]),
            ops.Y(circuit_1.qubits[1]),
            ops.RX(0.42, circuit_1.qubits[0]),
            ops.CX(circuit_1.qubits[0], circuit_1.qubits[1]),
        ]
        circuit_1.extend(operations)

        circuit_2 = Circuit(3)
        with circuit_2.context as regs:
            circuit_1((regs.q[0], regs.q[2]))

        assert circuit_2.circuit == [
            ops.X(circuit_2.qubits[0]),
            ops.Y(circuit_2.qubits[2]),
            ops.RX(0.42, circuit_2.qubits[0]),
            ops.CX(circuit_2.qubits[0], circuit_2.qubits[2]),
        ]

    def test_call_with_kwarg(self):
        """Test calling the circuit within a context to apply it to the active context
        when passing the qubits as a keyword argument."""
        circuit_1 = Circuit(2)
        operations = [
            ops.X(circuit_1.qubits[0]),
            ops.Y(circuit_1.qubits[1]),
            ops.RX(0.42, circuit_1.qubits[0]),
            ops.CX(circuit_1.qubits[0], circuit_1.qubits[1]),
        ]
        circuit_1.extend(operations)

        circuit_2 = Circuit(3)
        with circuit_2.context as regs:
            circuit_1(qubits=(regs.q[0], regs.q[2]))

        assert circuit_2.circuit == [
            ops.X(circuit_2.qubits[0]),
            ops.Y(circuit_2.qubits[2]),
            ops.RX(0.42, circuit_2.qubits[0]),
            ops.CX(circuit_2.qubits[0], circuit_2.qubits[2]),
        ]

    def test_call_single_qubit(self):
        """Test calling the circuit within a context to apply it to the active context
        when passing the qubits as a keyword argument."""
        circuit_1 = Circuit(1)
        operations = [ops.X(circuit_1.qubits[0]), ops.RY(0.42, circuit_1.qubits[0])]
        circuit_1.extend(operations)

        circuit_2 = Circuit(2)
        with circuit_2.context as regs:
            circuit_1(regs.q[1])

        assert circuit_2.circuit == [ops.X(circuit_2.qubits[1]), ops.RY(0.42, circuit_2.qubits[1])]

    def test_call_invalid_length(self):
        """Test that the correct exception is raised when calling a circuit with an incorrect number of qubits."""
        circuit_1 = Circuit(2)
        operations = [ops.X(circuit_1.qubits[0]), ops.RY(0.42, circuit_1.qubits[1])]
        circuit_1.extend(operations)

        circuit_2 = Circuit(2)
        with pytest.raises(ValueError, match="requires 2 qubits, got 1"):
            with circuit_2.context as regs:
                circuit_1(regs.q[1])

    def test_calling_in_own_context(self):
        """Test that the correct exception is raised when calling a circuit inside it's own context."""
        circuit_1 = Circuit(1)
        operations = [ops.X(circuit_1.qubits[0]), ops.RY(0.42, circuit_1.qubits[0])]
        circuit_1.extend(operations)

        with pytest.raises(TypeError, match="Cannot apply circuit in its own context."):
            with circuit_1.context as regs:
                circuit_1(regs.q[0])

    def test_call_outside_context(self):
        """Test that the correct exception is raised when calling a circuit outside of a context."""
        circuit_1 = Circuit(1)
        operations = [ops.X(circuit_1.qubits[0]), ops.RY(0.42, circuit_1.qubits[0])]
        circuit_1.extend(operations)

        with pytest.raises(
            CircuitError, match="Can only apply circuit object inside a circuit context."
        ):
            circuit_1(circuit_1.qubits[0])

    def test_call_bits(self):
        """Test calling the circuit with measurments and bits."""
        circuit_1 = Circuit(2, 2)
        operations = [
            ops.Hadamard(circuit_1.qubits[0]),
            ops.Hadamard(circuit_1.qubits[1]),
            ops.Measurement(circuit_1.qubits) | circuit_1.bits,
        ]
        circuit_1.extend(operations)

        circuit_2 = Circuit(3, 3)
        with circuit_2.context as regs:
            circuit_1((regs.q[0], regs.q[2]), (regs.c[0], regs.c[2]))

        assert circuit_2.circuit == [
            ops.Hadamard(circuit_2.qubits[0]),
            ops.Hadamard(circuit_2.qubits[2]),
            ops.Measurement((circuit_2.qubits[0], circuit_2.qubits[2]))
            | (circuit_2.bits[0], circuit_2.bits[2]),
        ]

    def test_call_single_bit(self):
        """Test calling the circuit within a context to apply it to the active context
        when passing a single bit."""
        circuit_1 = Circuit(1, 1)
        operations = [ops.Measurement(circuit_1.qubits) | circuit_1.bits]
        circuit_1.extend(operations)

        circuit_2 = Circuit(2, 2)
        with circuit_2.context as regs:
            circuit_1(regs.q[1], regs.c[1])

        assert circuit_2.circuit == [ops.Measurement(circuit_2.qubits[1]) | circuit_2.bits[1]]

    def test_call_bits_invalid_length(self):
        """Test that the correct exception is raised when calling a circuit
        with an incorrect number of bits."""
        circuit_1 = Circuit(2, 2)
        operations = [ops.Measurement(circuit_1.qubits) | circuit_1.bits]
        circuit_1.extend(operations)

        circuit_2 = Circuit(2, 2)
        with pytest.raises(ValueError, match="requires 2 bits, got 1"):
            with circuit_2.context as regs:
                circuit_1(regs.q, regs.c[1])

    def test_call_no_bits(self):
        """Test calling circuit with measurement but not passing bits."""
        circuit_1 = Circuit(2, 2)
        operations = [ops.Measurement(circuit_1.qubits) | circuit_1.bits]
        circuit_1.extend(operations)

        circuit_2 = Circuit(2, 2)
        with pytest.warns(UserWarning, match="Measurements not stored in circuit bits"):
            with circuit_2.context as regs:
                circuit_1(regs.q)

    def test_num_parameters_non_parametric_circuit(self):
        """Test that the ``num_parameters`` property returns the correct value when calling it on a
        non-parametric circuit."""
        circuit = Circuit(1)

        assert not circuit.parametric
        assert circuit.num_parameters == 0


class TestCircuitContext:
    """Unit tests for the CircuitContext class."""

    def test_initizialize_context(self, two_qubit_circuit):
        """Test initializing a ``CircuitContext``."""
        context = CircuitContext(two_qubit_circuit)

        assert context.circuit is two_qubit_circuit
        assert context.frozen is False
        assert context.active_context is None

    def test_frozen_context(self):
        """Test freezing a CircuitContext."""
        circuit = Circuit(2)
        context = CircuitContext(circuit)
        assert context.frozen is False

        with context:
            op_x = ops.X(circuit.qubits[0])

        assert op_x in circuit.circuit

        # unlock circuit since it's automatically locked when exiting connected context
        circuit.unlock()

        frozen_context = context.freeze

        with context:
            # active context ("context") should NOT be frozen
            assert context.frozen is False

            with frozen_context:
                # active context ("context") gets frozen within a frozen context ("frozen_context")
                assert context.frozen is True

                # operations will not be appended to frozen context
                op_y = ops.Y(circuit.qubits[0])

            # active context ("context") should unfreeze on frozen context ("frozen_context") exit
            assert context.frozen is False

        assert op_x in circuit.circuit
        assert op_y not in circuit.circuit

    def test_freezing_outside_context(self):
        """Test that the correct error is raised when using a frozen context
        outside of an active context."""
        circuit = Circuit(2)
        context = CircuitContext(circuit)
        assert context.frozen is False

        frozen_context = context.freeze

        with pytest.raises(CircuitError, match="Can only freeze active context."):
            with frozen_context:
                ops.Y(circuit.qubits[0])

    def test_locked_circuit(self):
        """Test that the correct error is raised when
        entering context with locked circuit."""
        circuit = Circuit(2)
        context = CircuitContext(circuit)
        assert context.circuit.is_locked() is False

        with context:
            ops.X(circuit.qubits[0])

        assert context.circuit.is_locked() is True

        with pytest.raises(CircuitError, match="Circuit is locked"):
            with context:
                ops.X(circuit.qubits[0])

    def test_nesting_contexts(self):
        """Test that the correct error is raised when nesting contexts."""
        circuit = Circuit(2)
        context = CircuitContext(circuit)

        with pytest.raises(RuntimeError, match="Cannot enter context"):
            with context:
                with context:
                    ops.X(circuit.qubits[0])

    def test_active_context(self):
        """Test the 'active_context' property."""
        circuit = Circuit(2)
        context = CircuitContext(circuit)

        assert context.active_context is None

        with context:
            assert context.active_context is context


class TestParametricCircuit:
    """Unit tests for the ParametricCircuit class."""

    def test_call_parametric_circuit(self):
        """Test calling a parametric circuit."""
        parametric_circuit = ParametricCircuit(1)
        circuit = Circuit(2)

        with parametric_circuit.context as regs:
            ops.X(regs.q[0])
            ops.RY(regs.p[0], regs.q[0])
            ops.RZ(3.3, regs.q[0])

        with circuit.context as regs:
            parametric_circuit([4.2], regs.q[1])

        assert circuit.circuit == [
            ops.X(circuit.qubits[1]),
            ops.RY(4.2, circuit.qubits[1]),
            ops.RZ(3.3, circuit.qubits[1]),
        ]

    def test_num_parameters(self):
        """Test that the ``num_parameters`` property returns the correct value."""
        parametric_circuit = ParametricCircuit(1)

        assert parametric_circuit.parametric is False
        assert parametric_circuit.num_parameters == 0

        with parametric_circuit.context as regs:
            ops.X(regs.q[0])
            ops.RY(regs.p[0], regs.q[0])
            ops.RZ(regs.p[1], regs.q[0])

        assert parametric_circuit.parametric is True
        assert parametric_circuit.num_parameters == 2

    def test_access_cached_context(self):
        """Test accessing a cached context by calling it twice."""
        parametric_circuit = ParametricCircuit(1)

        with parametric_circuit.context as regs:
            ops.RY(regs.p[0], regs.q[0])

        assert parametric_circuit.parametric is True
        assert parametric_circuit.num_parameters == 1

        parametric_circuit.unlock()
        with parametric_circuit.context as regs:
            ops.RZ(regs.p[1], regs.q[0])

        assert parametric_circuit.parametric is True
        assert parametric_circuit.num_parameters == 2

    def test_eval(self, empty_parametric_circuit):
        """Test evaluate circuit with parameters."""
        empty_parametric_circuit.add_qubit()
        with empty_parametric_circuit.context as regs:
            ops.RX(regs.p[0], regs.q[0])

        for op in empty_parametric_circuit.circuit:
            for p in op.parameters:
                assert isinstance(p, Variable)

        circuit = empty_parametric_circuit.eval([[4.2]], inplace=False)
        for op in circuit.circuit:
            assert op.parameters == [4.2]
            assert isinstance(op.parameters[0], float)

        for op in empty_parametric_circuit.circuit:
            assert isinstance(op.parameters[0], Variable)

    def test_eval_in_place(self, empty_parametric_circuit):
        """Test evaluate circuit in place with parameters."""
        empty_parametric_circuit.add_qubit()
        with empty_parametric_circuit.context as regs:
            ops.RX(regs.p[0], regs.q[0])

        for op in empty_parametric_circuit.circuit:
            for p in op.parameters:
                assert isinstance(p, Variable)

        empty_parametric_circuit.eval([[4.2]], inplace=True)
        for op in empty_parametric_circuit.circuit:
            assert op.parameters == [4.2]
            assert isinstance(op.parameters[0], float)

    def test_eval_no_params(self, empty_parametric_circuit):
        """Test that the correct exception is raised when evaluating circuit without parameters."""
        empty_parametric_circuit.add_qubit()
        with empty_parametric_circuit.context as regs:
            ops.RX(regs.p[0], regs.q[0])

        for op in empty_parametric_circuit.circuit:
            for p in op.parameters:
                assert isinstance(p, Variable)

        with pytest.raises(ValueError, match="No available parameter"):
            empty_parametric_circuit.eval()

    def test_call_outside_context(self):
        """Test that the correct exception is raised when calling a circuit outside of a context."""
        circuit_1 = ParametricCircuit(1)
        with circuit_1.context as regs:
            ops.RX(regs.p[0], regs.q[0])

        with pytest.raises(
            CircuitError, match="Can only apply circuit object inside a circuit context."
        ):
            circuit_1([0.42], circuit_1.qubits[0])


class TestParametricCircuitContext:
    """Unit tests for the ParametricCircuitContext class."""

    def test_initizialize_context_base_circuit(self, two_qubit_circuit):
        """Test that the correct exception is raised when initializing with a base circuit object."""
        with pytest.raises(
            TypeError, match="'ParametricCircuitContext' only works with 'ParametricCircuit'"
        ):
            ParametricCircuitContext(two_qubit_circuit)

    def test_initizialize_context(self, two_qubit_parametric_circuit):
        context = ParametricCircuitContext(two_qubit_parametric_circuit)

        assert context.circuit is two_qubit_parametric_circuit
        assert context.frozen is False
        assert context.active_context is None
