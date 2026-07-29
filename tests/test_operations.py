# Copyright 2026 D-Wave
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
from pytest_mock import MockerFixture

import dwave.gate.qcdl.implementations
from dwave.gate.qcdl import (
    QcdlModule,
    QCDLUserError,
    Register,
    Scope,
    operations,
    procedure,
    qcdl,
)
from dwave.gate.qcdl.operations import AngleType
from dwave.gate.qcdl.qcdl_models import QcdlStatement

available_operations = [
    name
    for name, obj in inspect.getmembers(operations, inspect.isfunction)
    if obj.__module__ == operations.__name__ and not name.startswith("__")
]


def test_basis_gates():
    basis_gates = {"rz", "x", "sx", "rzz", "cz"}
    assert basis_gates.issubset(set(available_operations))


@pytest.mark.parametrize("op_name", available_operations)
def test_operations(op_name):
    f = getattr(operations, op_name)
    sig = inspect.signature(f)
    num_qubits = sum(
        [
            1
            for parameter in sig.parameters.values()
            if parameter.annotation == QcdlModule
        ]
    )

    assert num_qubits in [1, 2]
    num_angles = sum(
        [
            1
            for parameter in sig.parameters.values()
            if parameter.annotation == AngleType
        ]
    )

    num_registers = sum(
        [
            1
            for parameter in sig.parameters.values()
            if parameter.annotation in [Register, Register | None]
        ]
    )

    if num_angles or num_registers:
        assert bool(num_angles) != bool(num_registers)

    if op_name in ["measure", "mced"]:
        assert num_registers == 1
    else:
        assert num_registers == 0

    @qcdl(5)
    def main(**kwargs):
        qubits = list(kwargs.values())
        sc = Scope(*qubits)
        angles = [0.1] * num_angles
        for idx in range(3):
            kwargs = (
                dict(register=sc.Register(name=f"reg{idx}"))
                if num_registers == 1
                else {}
            )
            f(*qubits[:num_qubits], *angles, **kwargs)

    qcdl_dict = main().model_dump(exclude_unset=True)

    for stmt in qcdl_dict["program"]["statements"]:
        statement = QcdlStatement.model_validate(stmt)
        if statement.op == op_name:
            break
        if op_name in ["sy", "sydg"]:
            assert statement.op == "ry"
            break
    else:
        raise ValueError(f"couldn't find {op_name}")
    assert statement.op == op_name or (
        op_name in ["sy", "sydg"] and statement.op == "ry"
    )
    assert [m.name for m in statement.modules if m.is_qubit] == [
        f"q{i}" for i in range(num_qubits)
    ]
    assert len(statement.non_module_args) == num_angles or (
        op_name in ["sy", "sydg"] and len(statement.non_module_args) == 1
    )
    if num_registers:
        assert "register" in statement.kwargs


@pytest.mark.parametrize("mirror", [True, False])
def test_measure_register(mirror, mocker: MockerFixture):
    @qcdl(10)
    def main(**kwargs):
        qubits = list(kwargs.values())
        scope = Scope(*qubits)
        register = scope.Register()
        operations.measure(qubit=qubits[0], register=register, mirror=mirror)

    spy = mocker.spy(dwave.gate.qcdl.implementations, "mirror_measurement_register")
    _ = main().model_dump(exclude_unset=True)
    assert spy.called == mirror


@pytest.mark.parametrize("mirror", [True, False])
def test_mced_register(mirror, mocker: MockerFixture):
    @qcdl(10)
    def main(**kwargs):
        qubits = list(kwargs.values())
        scope = Scope(*qubits)
        register = scope.Register()
        operations.mced(qubit=qubits[0], register=register, mirror=mirror)

    spy = mocker.spy(dwave.gate.qcdl.implementations, "mirror_bool_register")
    _ = main().model_dump(exclude_unset=True)
    assert spy.called == mirror
