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
import re
from contextlib import suppress
from operator import attrgetter
from typing import Callable

import pytest

from dwave.gate import Circuit
from dwave.gate import operations as ops

pyqir = pytest.importorskip("pyqir")
# QIR submodule imports must postcede PyQIR importorskip
from dwave.gate.qir.compiler import BaseModule, CompileError, qir_module  # noqa: E402
from dwave.gate.qir.instructions import InstrType, Instruction, Operations  # noqa: E402


@pytest.fixture(scope="function")
def two_op_module():
    """Module with an ``X``, a ``CNOT`` and a ``Measurement`` operation."""
    mod = BaseModule("QIR_module", 2, 1)

    args_op_0 = (mod.qubits[0],)
    args_op_1 = (mod.qubits[0], mod.qubits[1])
    args_meas = (mod.qubits[0], mod.results[0])

    instruction_0 = Instruction(InstrType.BASE, "x", args_op_0)
    instruction_1 = Instruction(InstrType.BASE, "cx", args_op_1)
    instruction_2 = Instruction(InstrType.MEASUREMENT, "mz", args_meas)

    mod.add_instruction("body", instruction_0)
    mod.add_instruction("body", instruction_1)
    mod.add_instruction("measurements", instruction_2)

    return mod


def assert_equal(str_0: str, str_1: str) -> bool:
    """Assert that two multi-line strings are equal apart from newlines
    and spaces padding lines."""

    def _strip(str_):
        for line in str_.splitlines():
            stripped = line.strip()
            if stripped != "":
                yield stripped

    strip_0 = _strip(str_0)
    strip_1 = _strip(str_1)
    while True:
        n1 = n2 = None
        with suppress(StopIteration):
            n1 = next(strip_0)
        with suppress(StopIteration):
            n2 = next(strip_1)

        if n1 is None and n2 is None:
            break

        assert n1 == n2


class TestBaseModule:
    """Unit tests for the ``BaseModule`` class."""

    @pytest.mark.parametrize(
        "type_name, pyqir_type, check, result",
        [
            ("VOID", pyqir.Type.void(pyqir.Context()), "is_void", True),
            ("DOUBLE", pyqir.Type.double(pyqir.Context()), "is_double", True),
            ("QUBIT", pyqir.qubit_type(pyqir.Context()), "pointee.name", "Qubit"),
            ("RESULT", pyqir.result_type(pyqir.Context()), "pointee.name", "Result"),
            ("INT32", pyqir.IntType(pyqir.Context(), 32), "width", 32),
            ("INT64", pyqir.IntType(pyqir.Context(), 64), "width", 64),
            ("INT_", pyqir.IntType(pyqir.Context(), 64), "width", 64),
            ("INT1", pyqir.IntType(pyqir.Context(), 1), "width", 1),
            ("BOOL_", pyqir.IntType(pyqir.Context(), 1), "width", 1),
        ],
    )
    def test_types_class(self, type_name, pyqir_type, check, result):
        """Test the internal ``BaseModule.Types`` dataclass."""
        mod = BaseModule("mod", 1, 1)

        assert attrgetter(f"{type_name}.{check}")(mod.Types) == result

    @pytest.mark.parametrize(
        "type_name, pyqir_type",
        [
            ("function", pyqir.FunctionType),
            ("pointer", pyqir.PointerType),
            ("ARRAY", pyqir.ArrayType),
            ("STRUCT", pyqir.StructType),
        ],
    )
    def test_callable_types_class(self, type_name, pyqir_type):
        """Test the internal ``BaseModule.Types`` dataclass functions."""
        mod = BaseModule("mod", 1, 1)

        assert isinstance(getattr(mod.Types, type_name), Callable)
        assert issubclass(getattr(mod.Types, type_name), pyqir_type)

    def test_basemodule(self):
        """Test the base module's main properties."""
        context = pyqir.Context()
        mod = BaseModule("mod_name", 2, 3, context=context)

        assert mod.name == "mod_name"

        assert len(mod.qubits) == 2
        assert all(q.type.pointee.name == "Qubit" for q in mod.qubits)

        assert len(mod.results) == 3
        assert all(r.type.pointee.name == "Result" for r in mod.results)

        assert mod.context == context

        assert isinstance(mod.builder, pyqir.Builder)

    def test_reset(self, two_op_module):
        """Test resetting the module keeping the instructions."""
        two_op_module.compile()

        assert two_op_module._instructions["body"]
        assert two_op_module._instructions["measurements"]
        assert two_op_module._is_compiled
        assert two_op_module._qir
        assert "__quantum__qis__x__body" in two_op_module._qir
        assert two_op_module._bitcode

        two_op_module.reset(rm_instructions=False)

        assert two_op_module._instructions["body"]
        assert two_op_module._instructions["measurements"]
        assert not two_op_module._is_compiled
        assert not two_op_module._qir
        assert not two_op_module._bitcode

        two_op_module.compile()

        assert two_op_module._is_compiled
        assert "__quantum__qis__x__body" in two_op_module._qir
        assert two_op_module._bitcode

    def test_reset_instructions(self, two_op_module):
        """Test resetting the module including removing instructions."""
        two_op_module.compile()

        assert two_op_module._instructions["body"]
        assert two_op_module._instructions["measurements"]
        assert two_op_module._is_compiled

        two_op_module.reset(rm_instructions=True)

        assert not two_op_module._instructions["body"]
        assert not two_op_module._instructions["measurements"]
        assert not two_op_module._is_compiled
        assert not two_op_module._qir
        assert not two_op_module._bitcode

        two_op_module.compile()

        assert two_op_module._is_compiled
        assert "__quantum__qis__x__body" not in two_op_module._qir
        assert two_op_module._bitcode

    @pytest.mark.parametrize("compile_first", [True, False])
    def test_qir(self, two_op_module, compile_first):
        """Test retrieving the QIR string."""
        assert "__quantum__qis__z__body" not in two_op_module.qir

        two_op_module.add_instruction(
            "body", Instruction(InstrType.BASE, "z", [two_op_module.qubits[0]])
        )

        two_op_module.reset(rm_instructions=False)

        if compile_first:
            two_op_module.compile()
        assert "__quantum__qis__z__body" in two_op_module.qir

    @pytest.mark.parametrize("compile_first", [True, False])
    def test_bitcode(self, two_op_module, compile_first):
        """Test retrieving the QIR bitcode."""
        two_op_module.add_instruction(
            "body", Instruction(InstrType.BASE, "z", [two_op_module.qubits[0]])
        )

        if compile_first:
            two_op_module.compile()

        qir_bitcode = two_op_module.bitcode

        assert qir_bitcode

    def test_external_void_return(self, two_op_module):
        """Test adding an external function to the module."""
        qb = two_op_module.Types.QUBIT
        two_op_module.add_external_function("banana", [qb, qb])

        instr = two_op_module.get_external_instruction("banana", two_op_module.qubits[:2])
        two_op_module.add_instruction("body", instr)

        assert "__quantum__qis__banana__body" in two_op_module.qir

        two_op_module.reset(rm_instructions=False)

        instr = two_op_module.get_external_instruction("banana", two_op_module.qubits[:2])
        two_op_module.add_instruction("body", instr)

        assert "__quantum__qis__banana__body" in two_op_module.qir

    def test_return(self, two_op_module):
        """Test setting a different return value than ``Void``."""

        two_op_module.add_instruction(
            "body", Instruction(InstrType.BASE, "z", [two_op_module.qubits[0]])
        )

        ret_value = two_op_module.results[0]
        two_op_module.set_return(ret_cmd=ret_value, force=True)

        # set strict to 'False' since other return values than
        # 'Void' otherwise would raise an exception during validation
        two_op_module.compile(strict=False)

        qir_str = two_op_module.qir

        assert f"ret {ret_value}" in qir_str

    def test_return_twice(self, two_op_module):
        """Test validation error."""

        two_op_module.add_instruction(
            "body", Instruction(InstrType.BASE, "z", [two_op_module.qubits[0]])
        )

        ret_value = two_op_module.results[0]
        two_op_module.set_return(ret_cmd=ret_value, force=True)

        with pytest.raises(CompileError):
            two_op_module.compile(strict=True)

    def test_return_twice_not_force(self, two_op_module):
        """Test that the correct error is raised when setting already set return."""

        two_op_module.add_instruction(
            "body", Instruction(InstrType.BASE, "z", [two_op_module.qubits[0]])
        )

        ret_value = two_op_module.results[0]
        two_op_module.set_return(ret_cmd=ret_value, force=False)

        with pytest.raises(ValueError, match=re.escape(f"Return {ret_value} already set.")):
            two_op_module.set_return(ret_cmd=ret_value, force=False)

    def test_compile_twice(self, two_op_module):
        """Test compiling a module twice."""

        two_op_module.add_instruction(
            "body", Instruction(InstrType.BASE, "z", [two_op_module.qubits[0]])
        )

        two_op_module.compile()

        with pytest.raises(CompileError, match="Module already compiled"):
            two_op_module.compile()

        two_op_module.compile(force=True)


class TestQirModuleFunction:
    """Tests for the ``qir_module``."""

    def test_compiling_circuit(self):
        """Test compiling a circuit."""
        circuit = Circuit(2, 2)

        with circuit.context as reg:
            ops.X(reg.q[0])
            ops.Y(reg.q[1])
            ops.Measurement(reg.q) | reg.c

        mod = qir_module(circuit)

        assert mod.name == "Circuit"

        assert len(mod.qubits) == 2
        assert len(mod.results) == 2

        x_instr = Instruction(InstrType.BASE, "x", [mod.qubits[0]])
        y_instr = Instruction(InstrType.BASE, "y", [mod.qubits[1]])
        assert mod._instructions["body"] == [x_instr, y_instr]

        q0_meas = Instruction(InstrType.MEASUREMENT, "mz", [mod.qubits[0], mod.results[0]])
        q1_meas = Instruction(InstrType.MEASUREMENT, "mz", [mod.qubits[1], mod.results[1]])
        assert mod._instructions["measurements"] == [q0_meas, q1_meas]

        r0_meas = Instruction(InstrType.OUTPUT, "out", [mod.results[0]])
        r1_meas = Instruction(InstrType.OUTPUT, "out", [mod.results[1]])
        assert mod._instructions["output"] == [r0_meas, r1_meas]

    def test_compile_circuit_with_parametric_op(self):
        """Test compiling a circuit with a parameteric operation."""
        circuit = Circuit(2)

        with circuit.context as reg:
            ops.X(reg.q[0])
            ops.RX(1.2, reg.q[1])

        mod = qir_module(circuit)

        assert len(mod.qubits) == 2
        assert len(mod.results) == 0

        x_instr = Instruction(InstrType.BASE, "x", [mod.qubits[0]])
        rx_instr = Instruction(InstrType.BASE, "rx", [1.2, mod.qubits[1]])
        assert mod._instructions["body"] == [x_instr, rx_instr]

    def test_mid_circuit_measurment(self):
        """Test that an error is raised when having mid-circuit measurments."""
        circuit = Circuit(2, 2)

        with circuit.context as reg:
            ops.X(reg.q[0])
            ops.Measurement(reg.q[0]) | reg.c[0]
            ops.Y(reg.q[1])

        with pytest.raises(
            ValueError, match="Mid-circuit measurements are not supported when compiling to QIR."
        ):
            _ = qir_module(circuit)

    # TODO: update when decompositions are implemented
    def test_op_decomposition_not_implemented(self, monkeypatch):
        """Test that an error is raised when attempting to decompose an
        operation which has no decompositiond defined."""
        circuit = Circuit(2, 2)

        with circuit.context as reg:
            ops.Rotation((0.1, 0.2, 0.3), reg.q[0])

        new_operations_to_qir = Operations.to_qir.copy()
        new_operations_to_qir.update({"Rotation": Operations.Op(InstrType.DECOMPOSE, "rot")})
        monkeypatch.setattr(Operations, "to_qir", new_operations_to_qir)

        assert Operations.to_qir["Rotation"].type == InstrType.DECOMPOSE

        with pytest.raises(NotImplementedError, match="Operation decompositions not supported."):
            _ = qir_module(circuit)

    # TODO: update when decompositions are implemented
    def test_op_decomposition_not_implemented_external(self, monkeypatch):
        """Test that an error is raised when attempting to decompose an
        external operation which has no decompositiond defined, when ``external == False``."""
        circuit = Circuit(2, 2)

        with circuit.context as reg:
            ops.Rotation((0.1, 0.2, 0.3), reg.q[0])

        assert Operations.to_qir["Rotation"].type == InstrType.EXTERNAL

        with pytest.raises(NotImplementedError, match="Operation decompositions not supported."):
            _ = qir_module(circuit)


class TestCircuitToQIR:
    """Tests for compiling a circuit into QIR."""

    def test_compile_simple_circuit(self):
        """Test compiling a circuit into QIR."""
        circuit = Circuit(2)

        with circuit.context as reg:
            ops.X(reg.q[0])
            ops.Y(reg.q[1])

        assert circuit.to_qir(bitcode=True)

        res = circuit.to_qir()
        exp = inspect.cleandoc(
            r"""
            ; ModuleID = 'Circuit'
            source_filename = "Circuit"

            %Qubit = type opaque

            define void @main() #0 {
              entry:
              call void @__quantum__rt__initialize(i8* null)
              br label %body

            body:                                             ; preds = %entry
              call void @__quantum__qis__x__body(%Qubit* null)
              call void @__quantum__qis__y__body(%Qubit* inttoptr (i64 1 to %Qubit*))
              br label %measurements

            measurements:                                     ; preds = %body
              br label %output

            output:                                           ; preds = %measurements
              ret void
            }

            declare void @__quantum__rt__initialize(i8*)

            declare void @__quantum__qis__x__body(%Qubit*)

            declare void @__quantum__qis__y__body(%Qubit*)

            attributes #0 = { "entry_point" "num_required_qubits"="2" "num_required_results"="0" "output_labeling_schema" "qir_profiles"="custom" }

            !llvm.module.flags = !{!0, !1, !2, !3}

            !0 = !{i32 1, !"qir_major_version", i32 1}
            !1 = !{i32 7, !"qir_minor_version", i32 0}
            !2 = !{i32 1, !"dynamic_qubit_management", i1 false}
            !3 = !{i32 1, !"dynamic_result_management", i1 false}
        """
        )

        assert_equal(res, exp)

    def test_compile_circuit_with_measurement(self):
        """Test compiling a circuit into QIR."""
        circuit = Circuit(3, 3)

        with circuit.context as reg:
            ops.X(reg.q[0])
            ops.CX(reg.q[1], reg.q[2])
            ops.Measurement(reg.q) | reg.c

        assert circuit.to_qir(add_external=True, bitcode=True)

        res = circuit.to_qir(add_external=True)
        exp = inspect.cleandoc(
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

        assert_equal(res, exp)

    def test_compile_circuit_with_external(self):
        """Test compiling a circuit with external operations into QIR."""
        circuit = Circuit(3, 2)

        with circuit.context as reg:
            ops.X(reg.q[0])
            ops.CY(reg.q[1], reg.q[2])

        assert circuit.to_qir(add_external=True, bitcode=True)

        res = circuit.to_qir(add_external=True)
        exp = inspect.cleandoc(
            r"""
            ; ModuleID = 'Circuit'
            source_filename = "Circuit"

            %Qubit = type opaque

            declare void @__quantum__qis__cy__body(%Qubit*, %Qubit*)

            define void @main() #0 {
            entry:
              call void @__quantum__rt__initialize(i8* null)
              br label %body

            body:                                             ; preds = %entry
              call void @__quantum__qis__x__body(%Qubit* null)
              call void @__quantum__qis__cy__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Qubit* inttoptr (i64 2 to %Qubit*))
              br label %measurements

            measurements:                                     ; preds = %body
              br label %output

            output:                                           ; preds = %measurements
              ret void
            }

            declare void @__quantum__rt__initialize(i8*)

            declare void @__quantum__qis__x__body(%Qubit*)

            attributes #0 = { "entry_point" "num_required_qubits"="3" "num_required_results"="2" "output_labeling_schema" "qir_profiles"="custom" }

            !llvm.module.flags = !{!0, !1, !2, !3}

            !0 = !{i32 1, !"qir_major_version", i32 1}
            !1 = !{i32 7, !"qir_minor_version", i32 0}
            !2 = !{i32 1, !"dynamic_qubit_management", i1 false}
            !3 = !{i32 1, !"dynamic_result_management", i1 false}
        """
        )

        assert_equal(res, exp)
