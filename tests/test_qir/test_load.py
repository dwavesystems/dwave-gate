# Copyright 2023 D-Wave Systems Inc.
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

import inspect

import pytest

from dwave.gate import operations as ops
from dwave.gate.circuit import Circuit
from dwave.gate.operations.base import Measurement, ParametricOperation

pyqir = pytest.importorskip("pyqir")
# QIR submodule imports must postcede PyQIR importorskip
from dwave.gate.qir.loader import load_qir_string  # noqa: E402


def assert_equal(circuit_0, circuit_1):
    """Checks two circuits for equality."""
    assert circuit_0.num_qubits == circuit_1.num_qubits
    assert circuit_0.num_bits == circuit_1.num_bits

    # check registers (but wait with values, since the qubits/bits will differ)
    assert circuit_0.qregisters.keys() == circuit_1.qregisters.keys()
    assert circuit_0.cregisters.keys() == circuit_1.cregisters.keys()

    # check qubit/bit labels (not the qubits/bits themselves since they will differ)
    qb_labels_0 = [qb.label for qb in circuit_0.qubits]
    qb_labels_1 = [qb.label for qb in circuit_1.qubits]
    assert qb_labels_0 == qb_labels_1

    mb_labels_0 = [mb.label for mb in circuit_0.bits]
    mb_labels_1 = [mb.label for mb in circuit_1.bits]
    assert mb_labels_0 == mb_labels_1

    # check parametricity of circuits
    assert circuit_0.parametric == circuit_1.parametric
    assert circuit_0.num_parameters == circuit_1.num_parameters

    # check number of operations
    assert len(circuit_0.circuit) == len(circuit_1.circuit)

    # check all operations; their names, qubits, bits and parameters (if required)
    for op_0, op_1 in zip(circuit_0.circuit, circuit_1.circuit):
        assert op_0.name == op_1.name
        assert type(op_0) == type(op_1)

        qb_labels_0 = [qb.label for qb in op_0.qubits]
        qb_labels_1 = [qb.label for qb in op_1.qubits]
        assert qb_labels_0 == qb_labels_1

        if isinstance(op_0, Measurement):
            mb_labels_0 = [mb.label for mb in op_0.bits]
            mb_labels_1 = [mb.label for mb in op_1.bits]
            assert mb_labels_0 == mb_labels_1

        if isinstance(op_0, ParametricOperation):
            assert op_0.parameters == op_1.parameters


class TestLoader:
    """Tests for loading QIR into a circuit."""

    def test_load_qir_string(self):
        """Test the ``load_qir_string`` function."""

        qir_string = inspect.cleandoc(
            r"""
            ; ModuleID = 'Circuit'
            source_filename = "Circuit"

            %Qubit = type opaque
            %Result = type opaque

            define void @main() #0 {
            entry:
              call void @__quantum__rt__initialize(i8* null)
              br label %body

            body:                                             ; preds = %entry
              call void @__quantum__qis__x__body(%Qubit* null)
              call void @__quantum__qis__ry__body(double 4.200000e+00, %Qubit* inttoptr (i64 2 to %Qubit*))
              call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Qubit* inttoptr (i64 2 to %Qubit*))
              br label %measurements

            measurements:                                     ; preds = %body
              call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
              call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* inttoptr (i64 1 to %Result*))
              call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 2 to %Qubit*), %Result* inttoptr (i64 2 to %Result*))
              br label %output

            output:                                           ; preds = %measurements
              call void @__quantum__rt__result_record_output(%Result* null, i8* null)
              call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 1 to %Result*), i8* null)
              call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 2 to %Result*), i8* null)
              ret void
            }

            declare void @__quantum__rt__initialize(i8*)

            declare void @__quantum__qis__x__body(%Qubit*)

            declare void @__quantum__qis__ry__body(double, %Qubit*)

            declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

            declare void @__quantum__qis__mz__body(%Qubit*, %Result* writeonly) #1

            declare void @__quantum__rt__result_record_output(%Result*, i8*)

            attributes #0 = { "entry_point" "num_required_qubits"="3" "num_required_results"="3" "output_labeling_schema" "qir_profiles"="custom" }
            attributes #1 = { "irreversible" }

            !llvm.module.flags = !{!0, !1, !2, !3}

            !0 = !{i32 1, !"qir_major_version", i32 1}
            !1 = !{i32 7, !"qir_minor_version", i32 0}
            !2 = !{i32 1, !"dynamic_qubit_management", i1 false}
            !3 = !{i32 1, !"dynamic_result_management", i1 false}
        """
        )

        circuit = load_qir_string(qir_string)

        assert circuit.num_qubits == 3
        assert circuit.num_bits == 3
        assert len(circuit.circuit) == 4

        assert circuit.circuit[0].name == "X"
        assert [qb.label for qb in circuit.circuit[0].qubits] == ["0"]

        assert circuit.circuit[1].name == "RY([4.2])"
        assert [qb.label for qb in circuit.circuit[1].qubits] == ["2"]
        assert circuit.circuit[1].parameters == [4.2]

        assert circuit.circuit[2].name == "CX"
        assert [qb.label for qb in circuit.circuit[2].qubits] == ["1", "2"]

        assert circuit.circuit[3].name == "Measurement"
        assert [mb.label for mb in circuit.circuit[3].bits] == ["0", "1", "2"]
        assert [qb.label for qb in circuit.circuit[3].qubits] == ["0", "1", "2"]

    @pytest.mark.parametrize("bitcode", [True, False])
    def test_to_from_qir(self, bitcode):
        """Test compiling and loading QIR works."""
        circuit = Circuit(3, 3)

        with circuit.context as reg:
            ops.X(reg.q[0])
            ops.CNOT(reg.q[2], reg.q[0])
            ops.RY(3.45, reg.q[1])
            ops.Measurement(reg.q) | reg.c

        qir = circuit.to_qir(bitcode=bitcode)
        circuit_from_qir = Circuit.from_qir(qir, bitcode=bitcode)

        assert_equal(circuit, circuit_from_qir)

    def test_invalid_qis_operation(self):
        """Test exception when using non-existent QIS operation."""

        qir_string = inspect.cleandoc(
            r"""
            ; ModuleID = 'Citrus'
            source_filename = "Citrus"

            %Qubit = type opaque

            define void @main() {
              entry:
              call void @__quantum__rt__initialize(i8* null)
              call void @__quantum__qis__xavier__body(%Qubit* null)
              ret void
            }

            declare void @__quantum__rt__initialize(i8*)
            declare void @__quantum__qis__xavier__body(%Qubit*)
        """
        )
        with pytest.raises(TypeError, match="not found in valid QIS operations"):
            _ = Circuit.from_qir(qir_string)

    def test_load_QIR_into_circuit(self):
        """Test loading QIR into an existing circuit."""
        qir_string = inspect.cleandoc(
            r"""
            ; ModuleID = 'Citrus'
            source_filename = "Citrus"

            %Qubit = type opaque

            define void @main() {
              entry:
              call void @__quantum__rt__initialize(i8* null)
              call void @__quantum__qis__y__body(%Qubit* null)
              ret void
            }

            declare void @__quantum__rt__initialize(i8*)
            declare void @__quantum__qis__y__body(%Qubit*)
        """
        )
        circuit = Circuit(2)
        circuit.append(ops.X(circuit.qubits[0]))

        assert len(circuit.circuit) == 1

        circuit = load_qir_string(qir_string, circuit=circuit)

        assert circuit.num_qubits == 2
        assert circuit.num_bits == 0
        assert len(circuit.circuit) == 2

        circuit.unlock()
        circuit.append(ops.Z(circuit.qubits[0]))

        assert [op.name for op in circuit.circuit] == ["X", "Y", "Z"]

    def test_early_ret(self):
        """Test setting a return in an earlier block."""
        qir_string = inspect.cleandoc(
            r"""
            ; ModuleID = 'Citrus'
            source_filename = "Citrus"

            %Qubit = type opaque

            define void @main() {
              entry:
              call void @__quantum__rt__initialize(i8* null)
              call void @__quantum__qis__x__body(%Qubit* null)
              ret void

              body:                                             ; preds = %entry
              call void @__quantum__qis__y__body(%Qubit* null)
              ret void
            }

            declare void @__quantum__rt__initialize(i8*)
            declare void @__quantum__qis__x__body(%Qubit*)
            declare void @__quantum__qis__y__body(%Qubit*)
        """
        )
        circuit = Circuit(2)

        circuit = load_qir_string(qir_string, circuit=circuit)

        assert len(circuit.circuit) == 1
        assert [op.name for op in circuit.circuit] == ["X"]
