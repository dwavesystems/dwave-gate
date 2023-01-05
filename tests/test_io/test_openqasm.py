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

from inspect import cleandoc

import pytest

import dwave.gate.operations as ops
from dwave.gate.circuit import Circuit, CircuitError, ParametricCircuit
from dwave.gate.operations.base import create_operation
from dwave.gate.primitives import Bit, Qubit
from dwave.gate.registers.registers import ClassicalRegister, QuantumRegister


class TestCircuitOpenQASM:
    """Unittests for generating OpenQASM 2.0 from a circuit."""

    def test_circuit(self):
        """Test generating OpenQASM 2.0 for a circuit."""
        circuit = Circuit(3)
        circuit.add_cregister(2, "apple")

        with circuit.context as regs:
            ops.X(regs.q[0])
            ops.RX(0.42, regs.q[1])
            ops.CNOT(regs.q[2], regs.q[0])

        assert circuit.to_qasm() == cleandoc(
            """
            OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[3];
            creg c[2];

            x q[0];
            rx(0.42) q[1];
            cx q[2], q[0];
        """
        )

    def test_circuit_several_qreg(self):
        """Test generating OpenQASM 2.0 for a circuit with several quantum registers."""
        circuit = Circuit(1)
        circuit.add_qregister(2, "apple")

        with circuit.context as regs:
            ops.X(regs.q[0])
            ops.RX(0.42, regs.q[1])
            ops.CNOT(regs.q[2], regs.q[0])

        assert circuit.to_qasm() == cleandoc(
            """
            OPENQASM 2.0;
            include "qelib1.inc";

            qreg q0[1];
            qreg q1[2];

            x q0[0];
            rx(0.42) q1[0];
            cx q1[1], q0[0];
        """
        )

    def test_circuit_several_creg(self):
        """Test generating OpenQASM 2.0 for a circuit with several classical registers."""
        circuit = Circuit(2)
        circuit.add_cregister(2, "apple")
        circuit.add_cregister(3, "banana")

        with circuit.context as regs:
            ops.X(regs.q[0])
            ops.RX(0.42, regs.q[1])
            ops.CNOT(regs.q[1], regs.q[0])

        assert circuit.to_qasm() == cleandoc(
            """
            OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[2];
            creg c0[2];
            creg c1[3];

            x q[0];
            rx(0.42) q[1];
            cx q[1], q[0];
        """
        )

    def test_circuit_reg_labels(self):
        """Test generating OpenQASM 2.0 for a circuit using reg labels."""
        circuit = Circuit(1)
        circuit.add_qregister(2, "apple")
        circuit.add_cregister(2, "banana")

        with circuit.context as regs:
            ops.X(regs.q[0])
            ops.RX(0.42, regs.q[1])
            ops.CNOT(regs.q[2], regs.q[0])

        assert circuit.to_qasm(reg_labels=True) == cleandoc(
            """
            OPENQASM 2.0;
            include "qelib1.inc";

            qreg qreg0[1];
            qreg apple[2];
            creg banana[2];

            x qreg0[0];
            rx(0.42) apple[0];
            cx apple[1], qreg0[0];
        """
        )

    def test_circuit_gate_definitions(self):
        """Test generating OpenQASM 2.0 for a circuit using gate definitions."""
        circuit = Circuit(3)

        with circuit.context as regs:
            ops.X(regs.q[0])
            ops.Rotation((0.1, 0.2, 0.3), regs.q[1])
            ops.CNOT(regs.q[2], regs.q[0])

        assert circuit.to_qasm(gate_definitions=True) == cleandoc(
            """
            OPENQASM 2.0;
            include "qelib1.inc";

            gate rot(beta, gamma, delta) { rz(beta) q[0]; ry(gamma) q[0]; rz(delta) q[0]; }

            qreg q[3];

            x q[0];
            rot(0.1, 0.2, 0.3) q[1];
            cx q[2], q[0];
        """
        )

    def test_parametric_circuit(self):
        """Test generating OpenQASM 2.0 for a parametric circuit."""

        circuit = ParametricCircuit(3)

        with circuit.context as regs:
            ops.X(regs.q[0])
            ops.RX(regs.p[0], regs.q[1])
            ops.CNOT(regs.q[2], regs.q[0])

        with pytest.raises(
            CircuitError, match="Parametric circuits cannot be transpiled into OpenQASM."
        ):
            circuit.to_qasm()


class TestOperationsOpenQASM:
    """Unittests for generating OpenQASM 2.0 from operations."""

    @pytest.mark.parametrize(
        "op, openqasm",
        [
            (ops.Identity(), "id"),
            (ops.X(), "x"),
            (ops.Y(), "y"),
            (ops.Z(), "z"),
            (ops.Hadamard(), "h"),
            (ops.T(), "t"),
            (ops.Phase(), "s"),
            (ops.RX(2), "rx(2)"),
            (ops.RY(2), "ry(2)"),
            (ops.RZ(2), "rz(2)"),
            (ops.Rotation((1, 2, 3)), "rot(1, 2, 3)"),
            (ops.CX(), "cx"),
            (ops.CY(), "cy"),
            (ops.CNOT(), "cx"),
            (ops.CZ(), "cz"),
            (ops.CHadamard(), "ch"),
            (ops.CRX(1), "crx(1)"),
            (ops.CRY(2), "cry(2)"),
            (ops.CRZ(4.2), "crz(4.2)"),
            (ops.CRotation((1, 2, 4.2)), "crot(1, 2, 4.2)"),
            (ops.SWAP(), "swap"),
            (ops.CSWAP(), "cswap"),
            (ops.CCNOT(), "ccx"),
        ],
    )
    def test_gate(self, op, openqasm):
        """Test generating OpenQASM 2.0 for a gate without qubits."""
        assert op.to_qasm() == openqasm

    @pytest.mark.parametrize(
        "op, openqasm",
        [
            (ops.Identity(Qubit(0)), "id q[0]"),
            (ops.X(Qubit(0)), "x q[0]"),
            (ops.Y(Qubit(0)), "y q[0]"),
            (ops.Z(Qubit(0)), "z q[0]"),
            (ops.Hadamard(Qubit(0)), "h q[0]"),
            (ops.T(Qubit(0)), "t q[0]"),
            (ops.Phase(Qubit(0)), "s q[0]"),
            (ops.RX(2, Qubit(0)), "rx(2) q[0]"),
            (ops.RY(2, Qubit(0)), "ry(2) q[0]"),
            (ops.RZ(2, Qubit(0)), "rz(2) q[0]"),
            (ops.Rotation((1, 2, 3), Qubit(0)), "rot(1, 2, 3) q[0]"),
            (ops.CX(Qubit(0), Qubit(1)), "cx q[0], q[1]"),
            (ops.CY(Qubit(0), Qubit(1)), "cy q[0], q[1]"),
            (ops.CNOT(Qubit(0), Qubit(1)), "cx q[0], q[1]"),
            (ops.CZ(Qubit(0), Qubit(1)), "cz q[0], q[1]"),
            (ops.CHadamard(Qubit(0), Qubit(1)), "ch q[0], q[1]"),
            (ops.CRX(1, Qubit(0), Qubit(1)), "crx(1) q[0], q[1]"),
            (ops.CRY(2, Qubit(0), Qubit(1)), "cry(2) q[0], q[1]"),
            (ops.CRZ(4.2, Qubit(0), Qubit(1)), "crz(4.2) q[0], q[1]"),
            (ops.CRotation((1, 2, 4.2), Qubit(0), Qubit(1)), "crot(1, 2, 4.2) q[0], q[1]"),
            (ops.SWAP((Qubit(0), Qubit(1))), "swap q[0], q[1]"),
            (ops.CSWAP((Qubit(0), Qubit(1), Qubit(2))), "cswap q[0], q[1], q[2]"),
            (ops.CCNOT((Qubit(0), Qubit(1), Qubit(2))), "ccx q[0], q[1], q[2]"),
        ],
    )
    def test_gate_qubits(selfop, op, openqasm):
        """Test generating OpenQASM 2.0 for a gate with qubits."""
        assert op.to_qasm() == openqasm

    def test_create_operation(self):
        """Test creating an operation."""
        circuit = Circuit(1)

        with circuit.context as regs:
            ops.Hadamard(regs.q[0])
            ops.X(regs.q[0])
            ops.Hadamard(regs.q[0])

        ZOp = create_operation(circuit, name="ZOp")

        assert ZOp().to_qasm() == "zop"
        assert ZOp(Qubit(0)).to_qasm() == "zop q[0]"

    def test_create_operation_no_label(self):
        """Test creating an operation without a label."""
        circuit = Circuit(1)

        with circuit.context as regs:
            ops.Hadamard(regs.q[0])
            ops.X(regs.q[0])
            ops.Hadamard(regs.q[0])

        ZOp = create_operation(circuit)

        assert ZOp().to_qasm() == "customoperation"
        assert ZOp(Qubit(0)).to_qasm() == "customoperation q[0]"


class TestRegisterOpenQASM:
    """Unittests for generating OpenQASM 2.0 from registers."""

    def test_gate_qreg(self):
        """Test generating OpenQASM 2.0 for a quantum register."""
        qreg = QuantumRegister()
        assert qreg.to_qasm() == "qreg q[0]"

    def test_gate_qreg_with_data(self):
        """Test generating OpenQASM 2.0 for a quantum register with data."""
        qreg = QuantumRegister([Qubit("banana"), Qubit("coconut")])
        assert qreg.to_qasm() == "qreg q[2]"

    def test_gate_creg(self):
        """Test generating OpenQASM 2.0 for a classical register."""
        qreg = ClassicalRegister()
        assert qreg.to_qasm() == "creg c[0]"

    def test_gate_creg_with_data(self):
        """Test generating OpenQASM 2.0 for a classical register with data."""
        qreg = ClassicalRegister([Bit("blueberry"), Bit("citrus")])
        assert qreg.to_qasm() == "creg c[2]"
