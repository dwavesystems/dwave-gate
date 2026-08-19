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
    QCDLModule,
    QCDLUserError,
    Register,
    Scope,
    operations,
    procedure,
    qcdl,
)
from dwave.gate.qcdl.operations import AngleType
from dwave.gate.qcdl.qcdl_models import QCDLStatement

available_operations = [
    name
    for name, obj in inspect.getmembers(operations, inspect.isfunction)
    if obj.__module__ == operations.__name__ and not name.startswith("_")
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
            if parameter.annotation == QCDLModule
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
        statement = QCDLStatement.model_validate(stmt)
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


def _count_annotated(op_name, annotation):
    parameters = inspect.signature(getattr(operations, op_name)).parameters.values()
    return sum(1 for parameter in parameters if parameter.annotation == annotation)


two_qubit_operations = [
    name
    for name in available_operations
    if _count_annotated(name, QCDLModule) == 2
]


def test_module_only_exports_operations():
    """``import *`` should bring in operations, not our imports."""
    namespace = {}
    exec("from dwave.gate.qcdl.operations import *", namespace)
    exported = set(namespace) - {"__builtins__"}

    assert exported == set(operations.__all__)
    assert set(available_operations) <= exported
    for leaked in ("np", "implementations", "inspect", "functools", "Sequence"):
        assert leaked not in exported


def test_all_lists_every_operation():
    assert sorted(operations.__all__) == sorted(available_operations + ["AngleType"])


def test_reset_is_an_importable_operation():
    """``q0.reset()`` works only through __getattr__, so it has no signature."""
    assert callable(operations.reset)
    assert operations.reset.__doc__
    assert list(inspect.signature(operations.reset).parameters) == ["qubit"]


def test_reset_matches_the_statement_getattr_produces():
    """The operation must be an alias for what the guide teaches, not a variant."""

    @qcdl(1)
    def with_operation(q0):
        operations.x(q0)
        operations.reset(q0)
        operations.measure(q0)

    @qcdl(1)
    def with_getattr(q0):
        operations.x(q0)
        q0.reset()
        operations.measure(q0)

    assert with_operation().model_dump() == with_getattr().model_dump()


@pytest.mark.parametrize("op_name", available_operations)
def test_operations_reject_a_non_qubit(op_name):
    """Without this the failure is an AttributeError naming ``procedure``."""
    f = getattr(operations, op_name)
    num_qubits = _count_annotated(op_name, QCDLModule)
    num_angles = _count_annotated(op_name, AngleType)
    needs_register = op_name in ("measure", "mced")

    @qcdl(2)
    def main(q0, q1):
        args = [5] + [q1] * (num_qubits - 1) + [0.1] * num_angles
        kwargs = dict(register=q0.Register(name="reg")) if needs_register else {}
        f(*args, **kwargs)

    with pytest.raises(QCDLUserError, match="must be a qubit"):
        main()


@pytest.mark.parametrize("value", [5, "q0", None, 3.14, [1]])
def test_non_qubit_error_names_the_parameter_and_the_type(value):
    @qcdl(1)
    def main(q0):
        operations.h(value)

    with pytest.raises(QCDLUserError) as cm:
        main()
    message = str(cm.value)
    assert "h() parameter 'qubit'" in message
    assert type(value).__name__ in message


@pytest.mark.parametrize("op_name", two_qubit_operations)
def test_two_qubit_operations_need_distinct_qubits(op_name):
    """``cx(q0, q0)`` used to build cleanly and fail only at the service."""
    f = getattr(operations, op_name)
    num_angles = _count_annotated(op_name, AngleType)

    @qcdl(2)
    def main(q0, q1):
        f(q0, q0, *[0.1] * num_angles)

    with pytest.raises(QCDLUserError, match="needs distinct qubits"):
        main()


@pytest.mark.parametrize("op_name", two_qubit_operations)
def test_two_qubit_operations_accept_distinct_qubits(op_name):
    f = getattr(operations, op_name)
    num_angles = _count_annotated(op_name, AngleType)

    @qcdl(2)
    def main(q0, q1):
        f(q0, q1, *[0.1] * num_angles)

    assert main().program.statements[0].op in (op_name, "ry")


def test_repeated_qubit_is_allowed_where_it_is_harmless():
    """``barrier``/``initialize`` take a set of qubits, so repeats are benign."""

    @qcdl(2)
    def main(q0, q1):
        operations.initialize(q0, q1, q0)
        operations.barrier(q0, q0)
        operations.measure(q0)

    assert [s.op for s in main().program.statements] == [
        "initialize",
        "barrier",
        "measure",
    ]


def test_validation_does_not_change_the_arity_error():
    """A wrong number of arguments still reports against the real signature."""

    @qcdl(1)
    def main(q0):
        operations.cx(q0)

    with pytest.raises(TypeError, match="target_qubit"):
        main()


def test_operations_keep_their_metadata():
    """The validation wrapper must not hide the signature or docs from sphinx."""
    for name in available_operations:
        f = getattr(operations, name)
        assert f.__name__ == name
        assert f.__doc__, name
        assert f.__module__ == operations.__name__


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
