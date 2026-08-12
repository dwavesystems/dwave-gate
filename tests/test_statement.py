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

import itertools

from dwave.gate.qcdl import QCDLModule, qcdl
from dwave.gate.qcdl.qcdl_models import QCDLStatement
from dwave.gate.qcdl.utils import is_qubit_name


def make_statement(qubit, op, *op_args, **op_kwargs):
    @qcdl(2)
    def main(q0: QCDLModule, q1: QCDLModule):
        qcdl_kwargs = dict(
            q0=q0,
            q1=q1,
            c0=q0.state.main.q("c0"),
            c1=q0.state.main.q("c1"),
        )
        q = qcdl_kwargs[qubit]
        args = [qcdl_kwargs.get(a, a) for a in op_args]
        kwargs = {k: qcdl_kwargs.get(v, v) for k, v in op_kwargs.items()}
        getattr(q, op)(*args, **kwargs)

    qcdl_json = main().model_dump(exclude_unset=True)
    return QCDLStatement.model_validate(qcdl_json["program"]["statements"][0])


def test_qubits_used():
    assert [m.name for m in make_statement("q0", "x").modules if m.is_qubit] == ["q0"]
    assert [m.name for m in make_statement("q0", "cx", "q1").modules if m.is_qubit] == [
        "q0",
        "q1",
    ]

    all_modules = ["q0", "q1", "c0", "c1"]
    op_name = "my_op"
    for num_mods in range(1, 5):
        for use_mods in itertools.combinations(all_modules, num_mods):
            stmt = make_statement(use_mods[0], "my_op", *use_mods[1:])
            assert not stmt.is_procedure_call
            assert op_name in str(stmt)
            for name in use_mods:
                assert name in str(stmt)

            assert sorted(m.name for m in stmt.modules) == sorted(use_mods)
            assert len(stmt.qargs) == len([m for m in stmt.modules if m.is_qubit])
            qargs = [int(q[1:]) for q in use_mods if is_qubit_name(q)]
            assert stmt.qargs == qargs


def test_conditionals():
    for op in ["_If", "c_if", "comment"]:
        assert [
            m.name for m in make_statement("q0", op, "q1").modules if m.is_qubit
        ] == ["q0"]
        assert [
            m.name for m in make_statement("q1", op, "q0").modules if m.is_qubit
        ] == ["q1"]

    assert make_statement("q1", "c_if", "x", "q0").condition == "q0"
    assert make_statement("q1", "_If", "q0").condition == "q0"
    assert make_statement("q1", "_If", condition="q0").condition == "q0"


def test_statement_model_validates_dict():
    """QCDLStatement can be validated from a raw dict."""
    stmt = QCDLStatement.model_validate({"op": "cx", "qubit": "q0", "args": ["q1"]})
    assert stmt.op == "cx"
    assert stmt.qubit_name == "q0"
    assert any(m.name == "q0" for m in stmt.modules)
    assert any(m.name == "q1" for m in stmt.modules)


def test_simple_desc():
    for op in [
        ("_If", "If(q1)"),
        ("c_if", "q0.c_if(q1)"),
        ("comment", "q0.comment(q1)"),
    ]:
        assert make_statement("q0", op[0], "q1").simple_desc() == op[1]
