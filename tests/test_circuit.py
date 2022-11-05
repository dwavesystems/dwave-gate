# Confidential & Proprietary Information: D-Wave Systems Inc.
import pytest

import dwave.gate.operations as ops
from dwave.gate import CircuitContext
from dwave.gate.circuit import Circuit, CircuitError, ParametricCircuit, ParametricCircuitContext
from dwave.gate.primitives import Bit, Qubit
from dwave.gate.registers import ClassicalRegister, QuantumRegister


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

        circuit.append(operations)

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

    def test_add_qubit(self, two_qubit_circuit):
        """Test adding qubits to a circuit."""
        assert two_qubit_circuit.num_qubits == 2
        assert [qb.label for qb in two_qubit_circuit.qubits] == ["0", "1"]

        two_qubit_circuit.add_qubit(Qubit("pineapple"))
        two_qubit_circuit.add_qubit()

        assert two_qubit_circuit.num_qubits == 4
        assert [qb.label for qb in two_qubit_circuit.qubits] == ["0", "1", "pineapple", "3"]

    def test_add_qubit_to_empty_circuit(self, empty_circuit):
        """Test adding qubits to an empty circuit."""
        assert empty_circuit.num_qubits == 0
        assert empty_circuit.qubits == []

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

    def test_add_bit_to_empty_circuit(self, empty_circuit):
        """Test adding bits to a circuit."""
        assert empty_circuit.num_bits == 0
        assert empty_circuit.bits == []

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

    def test_call(self):
        """Test calling the circuit within a context to apply it to the active context."""
        circuit_1 = Circuit(2)
        operations = [
            ops.X(circuit_1.qubits[0]),
            ops.Y(circuit_1.qubits[1]),
            ops.RX(0.42, circuit_1.qubits[0]),
            ops.CX(circuit_1.qubits[0], circuit_1.qubits[1]),
        ]
        circuit_1.append(operations)

        circuit_2 = Circuit(3)
        with circuit_2.context as q:
            circuit_1((q[0], q[2]))

        assert circuit_2.circuit == [
            ops.X(circuit_2.qubits[0]),
            ops.Y(circuit_2.qubits[2]),
            ops.RX(0.42, circuit_2.qubits[0]),
            ops.CX(circuit_2.qubits[0], circuit_2.qubits[2]),
        ]

    def test_call_no_args(self):
        """Test that the correct exception is raised when calling the circuit within a context
        without any arguments."""
        circuit_1 = Circuit(2)
        circuit_1.append(ops.X(circuit_1.qubits[0]))

        circuit_2 = Circuit(3)
        with pytest.raises(ValueError, match="Circuit requires 2 qubits, got 0."):
            with circuit_2.context:
                circuit_1()

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
        circuit_1.append(operations)

        circuit_2 = Circuit(3)
        with circuit_2.context as q:
            circuit_1(qubits=(q[0], q[2]))

        assert circuit_2.circuit == [
            ops.X(circuit_2.qubits[0]),
            ops.Y(circuit_2.qubits[2]),
            ops.RX(0.42, circuit_2.qubits[0]),
            ops.CX(circuit_2.qubits[0], circuit_2.qubits[2]),
        ]

    def test_call_with_invalid_kwarg(self):
        """Test calling the circuit with an invalid kwarg."""
        circuit_1 = Circuit(2)
        operations = [
            ops.X(circuit_1.qubits[0]),
            ops.Y(circuit_1.qubits[1]),
            ops.RX(0.42, circuit_1.qubits[0]),
            ops.CX(circuit_1.qubits[0], circuit_1.qubits[1]),
        ]
        circuit_1.append(operations)

        circuit_2 = Circuit(3)

        with pytest.raises(TypeError, match="got unexpected keyword"):
            with circuit_2.context as q:
                circuit_1(peabits=(q[0], q[2]))

    def test_call_with_too_many_args(self):
        """Test calling the circuit with too many arguments."""
        circuit_1 = Circuit(2)
        operations = [
            ops.X(circuit_1.qubits[0]),
            ops.Y(circuit_1.qubits[1]),
            ops.RX(0.42, circuit_1.qubits[0]),
            ops.CX(circuit_1.qubits[0], circuit_1.qubits[1]),
        ]
        circuit_1.append(operations)

        circuit_2 = Circuit(3)

        with pytest.raises(TypeError, match="takes from 1 to 2 arguments"):
            with circuit_2.context as q:
                circuit_1(31, q[0], q[2])

    def test_call_single_qubit(self):
        """Test calling the circuit within a context to apply it to the active context
        when passing the qubits as a keyword argument."""
        circuit_1 = Circuit(1)
        operations = [ops.X(circuit_1.qubits[0]), ops.RY(0.42, circuit_1.qubits[0])]
        circuit_1.append(operations)

        circuit_2 = Circuit(2)
        with circuit_2.context as q:
            circuit_1(q[1])

        assert circuit_2.circuit == [ops.X(circuit_2.qubits[1]), ops.RY(0.42, circuit_2.qubits[1])]

    def test_call_invalid_length(self):
        """Test that the correct exception is raised when calling a circuit with an incorrect number of qubits."""
        circuit_1 = Circuit(2)
        operations = [ops.X(circuit_1.qubits[0]), ops.RY(0.42, circuit_1.qubits[1])]
        circuit_1.append(operations)

        circuit_2 = Circuit(2)
        with pytest.raises(ValueError, match="requires 2 qubits, got 1"):
            with circuit_2.context as q:
                circuit_1(q[1])

    def test_calling_in_own_context(self):
        """Test that the correct exception is raised when calling a circuit inside it's own context."""
        circuit_1 = Circuit(1)
        operations = [ops.X(circuit_1.qubits[0]), ops.RY(0.42, circuit_1.qubits[0])]
        circuit_1.append(operations)

        with pytest.raises(TypeError, match="Cannot apply circuit in its own context."):
            with circuit_1.context as q:
                circuit_1(q[0])

    def test_call_outside_context(self):
        """Test that the correct exception is raised when calling a circuit outside of a context."""
        circuit_1 = Circuit(1)
        operations = [ops.X(circuit_1.qubits[0]), ops.RY(0.42, circuit_1.qubits[0])]
        circuit_1.append(operations)

        with pytest.raises(
            CircuitError, match="Can only apply circuit object inside a circuit context."
        ):
            circuit_1(circuit_1.qubits[0])

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

        with parametric_circuit.context as (p, q):
            ops.X(q[0])
            ops.RY(p[0], q[0])
            ops.RZ(3.3, q[0])

        with circuit.context as q:
            parametric_circuit([4.2], q[1])

        assert circuit.circuit == [
            ops.X(circuit.qubits[1]),
            ops.RY(4.2, circuit.qubits[1]),
            ops.RZ(3.3, circuit.qubits[1]),
        ]

    def test_call_parametric_circuit_with_kwarg(self):
        """Test calling a parametric circuit passing parameters as a kwarg."""
        parametric_circuit = ParametricCircuit(1)
        circuit = Circuit(2)

        with parametric_circuit.context as (p, q):
            ops.X(q[0])
            ops.RY(p[0], q[0])
            ops.RZ(3.3, q[0])

        with circuit.context as q:
            parametric_circuit(q[1], parameters=[4.2])

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

        with parametric_circuit.context as (p, q):
            ops.X(q[0])
            ops.RY(p[0], q[0])
            ops.RZ(p[1], q[0])

        assert parametric_circuit.parametric is True
        assert parametric_circuit.num_parameters == 2

    def test_access_cached_context(self):
        """Test accessing a cached context by calling it twice."""
        parametric_circuit = ParametricCircuit(1)

        with parametric_circuit.context as (p, q):
            ops.RY(p[0], q[0])

        assert parametric_circuit.parametric is True
        assert parametric_circuit.num_parameters == 1

        parametric_circuit.unlock()
        with parametric_circuit.context as (p, q):
            ops.RZ(p[1], q[0])

        assert parametric_circuit.parametric is True
        assert parametric_circuit.num_parameters == 2


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
