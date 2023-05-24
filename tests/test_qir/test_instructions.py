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

import itertools

import pytest

pyqir = pytest.importorskip("pyqir")
# QIR submodule imports must postcede PyQIR importorskip
from dwave.gate.qir.compiler import BaseModule  # noqa: E402
from dwave.gate.qir.instructions import InstrType, Instruction  # noqa: E402


class TestInstruction:
    """Tests for the ``Instruction`` class."""

    @pytest.mark.parametrize(
        "type_name", ["base", "external", "decompose", "invalid", "skip", "measurement", "output"]
    )
    def test_instruction(self, type_name, dummy_function):
        """Test the ``Instruction`` initializer."""
        name = "pomelo"
        args = [1.2, 2.3]
        type_ = getattr(InstrType, type_name)
        instr = Instruction(type_, name, args, external=dummy_function)

        assert instr.name == name
        assert instr.type is type_
        assert instr.args == args

    @pytest.mark.parametrize("args", [[1 + 2j], [0.1, "hello"]])
    def test_args_error(self, args):
        """Test that the correct exception is raised when passing invalid argument types."""

        with pytest.raises(TypeError, match="Incorrect type"):
            Instruction(InstrType.base, "pomelo", args)

    @pytest.mark.parametrize("type_name", ["base", "measurement"])
    def test_excecute_instruction(self, type_name, mocker):
        """Test excecuting instructions."""
        type_ = getattr(InstrType, type_name)
        instr = Instruction(type_, "pomelo", [])

        mod = BaseModule("QIR_module", 2, 1)

        def _assert_qis_called():
            assert instr.type in (InstrType.base, InstrType.measurement)

        module = Instruction.__module__
        qis_mock = mocker.patch(f"{module}.Instruction._execute_qis")
        qis_mock.side_effect = lambda _: _assert_qis_called()

        instr.execute(mod.builder)

        qis_mock.assert_called_once()

    def test_excecute_instruction_ext(self, mocker, dummy_function):
        """Test excecuting instructions."""
        instr = Instruction(InstrType.external, "pomelo", [], external=dummy_function)

        mod = BaseModule("QIR_module", 2, 1)

        def _assert_ext_called():
            assert instr.type is InstrType.external

        module = Instruction.__module__
        ext_mock = mocker.patch(f"{module}.Instruction._execute_external")
        ext_mock.side_effect = lambda _: _assert_ext_called()

        instr.execute(mod.builder)

        ext_mock.assert_called_once()

    def test_instruction_no_external(
        self,
    ):
        """Test external instruction with no external function."""

        with pytest.raises(
            ValueError, match="Instruction with type 'external' missing external function."
        ):
            _ = Instruction(InstrType.external, "pomelo", [])

    @pytest.mark.parametrize("type_name", ["decompose", "invalid", "skip", "output"])
    def test_excecute_instruction_error(self, type_name, monkeypatch):
        """Assert that the correct exception is raised when attempting
        to excecute a non-excecutable instructions."""
        args = [1.2, 2.3]
        type_ = getattr(InstrType, type_name)
        instr = Instruction(type_, "pomelo", args)

        mod = BaseModule("QIR_module", 2, 1)

        with pytest.raises(TypeError, match="Cannot execute instruction"):
            instr.execute(mod.builder)

    def test_instruction_equality(self, dummy_function):
        """Test equality between ``Instruction`` objects."""

        instr_0 = Instruction(InstrType.base, "pomelo", [1.2, 2.3])
        instr_1 = Instruction(InstrType.external, "pomelo", [1.2, 2.3], external=dummy_function)
        instr_2 = Instruction(InstrType.base, "grapefruit", [1.2, 2.3])
        instr_3 = Instruction(InstrType.base, "pomelo", [9, 8, 7, 6])

        for i0, i1 in itertools.combinations([instr_0, instr_1, instr_2, instr_3], r=2):
            assert i0 != i1

        instr_0_dup = Instruction(InstrType.base, "pomelo", [1.2, 2.3])
        instr_3_dup = Instruction(InstrType.base, "pomelo", [9.0, 8.0, 7.0, 6.0])

        assert instr_0 == instr_0_dup
        assert instr_3 == instr_3_dup
