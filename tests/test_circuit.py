# Confidential & Proprietary Information: D-Wave Systems Inc.
import pytest

import dwgms.operations as ops
from dwgms import CircuitContext
from dwgms.circuit import Circuit, CircuitError
from dwgms.registers import ClassicalRegister, QuantumRegister


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

        # check (qu)bits in the circuit
        assert circuit.qubits == [f"q{i}" for i in range(qubits)]
        assert circuit.bits == [f"c{i}" for i in range(bits)]

        # check registers in the circuit
        assert isinstance(circuit.qregisters["qreg0"], QuantumRegister)
        assert isinstance(circuit.cregisters["creg0"], ClassicalRegister)

    def test_qubits(self):
        """Test circuit with qubits."""
        circuit = Circuit(num_qubits=4)

        assert circuit.num_qubits == 4
        assert circuit.num_bits == 0

        assert circuit.qubits == ["q0", "q1", "q2", "q3"]
        assert isinstance(circuit.qregisters["qreg0"], QuantumRegister)

    def test_bits(self):
        """Test circuit with bits."""
        circuit = Circuit(num_bits=3)

        assert circuit.num_qubits == 0
        assert circuit.num_bits == 3

        assert circuit.bits == ["c0", "c1", "c2"]
        assert isinstance(circuit.cregisters["creg0"], ClassicalRegister)

    def test_representation(self):
        """Test represenation of a circuit."""
        circuit = Circuit(2)
        assert circuit.__repr__() == f"<Circuit: qubits=2, bits=0, ops=0>"

    def test_appending_operations(self):
        """Test adding operations to a circuit."""
        circuit = Circuit(2)
        operations = [ops.X("q0"), ops.Y("q1"), ops.CX("q0", "q1")]

        circuit.append(operations)

        assert circuit.circuit == operations

    def test_appending_operations_locked(self, two_qubit_circuit):
        """Test adding operations to a locked circuit."""
        op = ops.X("q0")

        two_qubit_circuit.lock()
        with pytest.raises(CircuitError, match="Circuit is locked"):
            two_qubit_circuit.append(op)

        two_qubit_circuit.unlock()
        two_qubit_circuit.append(op)
        assert op in two_qubit_circuit.circuit

    def test_appending_operations_invalid_qubits(self):
        """Test adding operations on invalid qubits to a circuit."""
        circuit = Circuit(2)

        with pytest.raises(ValueError, match="Qubit 'coconut' not in circuit."):
            circuit.append(ops.X("coconut"))

    def test_remove(self):
        """Test removing an operation from a circuit."""
        circuit = Circuit(2)

        # add two operations to make sure only a single op can be removed
        op = ops.X("q0")
        op_extra = ops.CNOT("q0", "q1")

        circuit.append(op)
        circuit.append(op_extra)
        assert circuit.circuit == [op, op_extra]

        circuit.remove(op)
        assert circuit.circuit == [op_extra]

    def test_invalid_remove(self):
        """Test removing an operation which is not part of the circuit."""
        circuit = Circuit(2)
        op = ops.X("q0")

        circuit.append(op)
        assert circuit.circuit == [op]

        with pytest.raises(ValueError, match="not in circuit"):
            circuit.remove(ops.Y("q0"))

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

        assert two_qubit_circuit.qregisters is not None
        assert two_qubit_circuit.cregisters is not None

        two_qubit_circuit.reset(keep_registers=keep_registers)

        assert two_qubit_circuit.circuit == []
        assert context is not two_qubit_circuit.context

        if keep_registers:
            assert two_qubit_circuit.qregisters is not None
            assert two_qubit_circuit.cregisters is not None
        else:
            assert two_qubit_circuit.qregisters is None
            assert two_qubit_circuit.cregisters is None

    def test_add_qubit(self, two_qubit_circuit):
        """Test adding qubits to a circuit."""
        assert two_qubit_circuit.num_qubits == 2
        assert two_qubit_circuit.qubits == ["q0", "q1"]

        two_qubit_circuit.add_qubit("pineapple")
        two_qubit_circuit.add_qubit()

        assert two_qubit_circuit.num_qubits == 4
        assert two_qubit_circuit.qubits == ["q0", "q1", "pineapple", "q3"]

    def test_add_qubit_to_empty_circuit(self, empty_circuit):
        """Test adding qubits to an empty circuit."""
        assert empty_circuit.num_qubits == 0
        assert empty_circuit.qubits == []

        empty_circuit.add_qubit("pineapple")
        empty_circuit.add_qubit()

        assert empty_circuit.num_qubits == 2
        assert empty_circuit.qubits == ["pineapple", "q1"]

    def test_add_qubit_with_qreg_label(self, two_qubit_circuit):
        """Test adding qubits to a circuit specifying a qreg label."""
        assert two_qubit_circuit.num_qubits == 2
        assert two_qubit_circuit.qubits == ["q0", "q1"]

        # add a qubit to a non-existing qreg
        two_qubit_circuit.add_qubit("pineapple", qreg_label="coconut")
        # add a qubit to an already existing qreg (default name)
        two_qubit_circuit.add_qubit("crabapple", qreg_label="qreg0")

        assert two_qubit_circuit.num_qubits == 4
        # the order of qubits depend on the order of the qregs
        # 'qreg0' qubits first, and then 'coconut' qubits
        assert two_qubit_circuit.qubits == ["q0", "q1", "crabapple", "pineapple"]
        assert two_qubit_circuit.qregisters["coconut"].data == ["pineapple"]
        assert two_qubit_circuit.qregisters["qreg0"].data == ["q0", "q1", "crabapple"]

    @pytest.mark.parametrize("qreg_label", [None, "qreg0", "coconut"])
    def test_add_qubit_with_existing_name(self, two_qubit_circuit, qreg_label):
        """Test adding a qubit to a circuit using an already existing label."""
        assert "q0" in two_qubit_circuit.qregisters["qreg0"].data

        with pytest.raises(ValueError, match="already in use"):
            two_qubit_circuit.add_qubit("q0", qreg_label=qreg_label)

    def test_add_bit(self, two_bit_circuit):
        """Test adding bits to a circuit."""
        assert two_bit_circuit.num_bits == 2
        assert two_bit_circuit.bits == ["c0", "c1"]

        two_bit_circuit.add_bit("pineapple")
        two_bit_circuit.add_bit()

        assert two_bit_circuit.num_bits == 4
        assert two_bit_circuit.bits == ["c0", "c1", "pineapple", "c3"]

    def test_add_bit_to_empty_circuit(self, empty_circuit):
        """Test adding bits to a circuit."""
        assert empty_circuit.num_bits == 0
        assert empty_circuit.bits == []

        empty_circuit.add_bit("pineapple")
        empty_circuit.add_bit()

        assert empty_circuit.num_bits == 2
        assert empty_circuit.bits == ["pineapple", "c1"]

    def test_add_bit_with_creg_label(self, two_bit_circuit):
        """Test adding bits to a circuit specifying a creg label."""
        assert two_bit_circuit.num_bits == 2
        assert two_bit_circuit.bits == ["c0", "c1"]

        # add a bit to a non-existing creg
        two_bit_circuit.add_bit("pineapple", creg_label="coconut")
        # add a bit to an already existing creg (default name)
        two_bit_circuit.add_bit("crabapple", creg_label="creg0")

        assert two_bit_circuit.num_bits == 4
        # the order of qubits depend on the order of the qregs
        # 'creg0' bits first, and then 'coconut' bits
        assert two_bit_circuit.bits == ["c0", "c1", "crabapple", "pineapple"]
        assert two_bit_circuit.cregisters["coconut"].data == ["pineapple"]
        assert two_bit_circuit.cregisters["creg0"].data == ["c0", "c1", "crabapple"]

    @pytest.mark.parametrize("creg_label", [None, "creg0", "coconut"])
    def test_add_bit_with_existing_label(self, two_bit_circuit, creg_label):
        """Test adding a bit to a circuit using an already existing label."""
        assert "c0" in two_bit_circuit.cregisters["creg0"].data

        with pytest.raises(ValueError, match="already in use"):
            two_bit_circuit.add_bit("c0", creg_label=creg_label)

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

    def test_call(self):
        """Test calling the circuit within a context to apply it to the active context."""
        circuit_1 = Circuit(2)
        operations = [ops.X("q0"), ops.Y("q1"), ops.RX(0.42, "q0"), ops.CX("q0", "q1")]
        circuit_1.append(operations)

        circuit_2 = Circuit(3)
        with circuit_2.context as q:
            circuit_1(q[0], q[2])

        assert circuit_2.circuit == [
            ops.X("q0"),
            ops.Y("q2"),
            ops.RX(0.42, "q0"),
            ops.CX("q0", "q2"),
        ]

    def test_call_with_kwarg(self):
        """Test calling the circuit within a context to apply it to the active context
        when passing the qubits as a keyword argument."""
        circuit_1 = Circuit(2)
        operations = [ops.X("q0"), ops.Y("q1"), ops.RX(0.42, "q0"), ops.CX("q0", "q1")]
        circuit_1.append(operations)

        circuit_2 = Circuit(3)
        with circuit_2.context as q:
            circuit_1(qubits=(q[0], q[2]))

        assert circuit_2.circuit == [
            ops.X("q0"),
            ops.Y("q2"),
            ops.RX(0.42, "q0"),
            ops.CX("q0", "q2"),
        ]

    def test_call_single_qubit(self):
        """Test calling the circuit within a context to apply it to the active context
        when passing the qubits as a keyword argument."""
        circuit_1 = Circuit(1)
        operations = [ops.X("q0"), ops.RY(0.42, "q0")]
        circuit_1.append(operations)

        circuit_2 = Circuit(2)
        with circuit_2.context as q:
            circuit_1(q[1])

        assert circuit_2.circuit == [ops.X("q1"), ops.RY(0.42, "q1")]

    def test_call_invalid_length(self):
        """Test that the correct exception is raised when calling a circuit with an incorrect number of qubits."""
        circuit_1 = Circuit(2)
        operations = [ops.X("q0"), ops.RY(0.42, "q1")]
        circuit_1.append(operations)

        circuit_2 = Circuit(2)
        with pytest.raises(ValueError, match="requires 2 qubits, got 1"):
            with circuit_2.context as q:
                circuit_1(q[1])

    def test_call_outside_context(self):
        """Test that the correct exception is raised when calling a circuit outside of a context."""
        circuit_1 = Circuit(1)
        operations = [ops.X("q0"), ops.RY(0.42, "q0")]
        circuit_1.append(operations)

        with pytest.raises(
            CircuitError, match="Can only apply circuit object inside a circuit context."
        ):
            circuit_1("q0")


class TestCircuitContext:
    """Unit tests for the CircuitContext class."""

    def test_initizialize_context(self, two_qubit_circuit):
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
            op_x = ops.X("q0")

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
                op_y = ops.Y("q0")

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
                ops.Y("q0")

    def test_locked_circuit(self):
        """Test that the correct error is raised when
        entering context with locked circuit."""
        circuit = Circuit(2)
        context = CircuitContext(circuit)
        assert context.circuit.is_locked() is False

        with context:
            ops.X("q0")

        assert context.circuit.is_locked() is True

        with pytest.raises(CircuitError, match="Circuit is locked"):
            with context:
                ops.X("q0")

    def test_nesting_contexts(self):
        """Test that the correct error is raised when nesting contexts."""
        circuit = Circuit(2)
        context = CircuitContext(circuit)

        with pytest.raises(RuntimeError, match="Cannot enter context"):
            with context:
                with context:
                    ops.X("q0")

    def test_active_context(self):
        """Test the 'active_context' property."""
        circuit = Circuit(2)
        context = CircuitContext(circuit)

        assert context.active_context is None

        with context:
            assert context.active_context is context
