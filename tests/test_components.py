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

import json
import pickle

import numpy as np
import pytest

from dwave.gate.qcdl import (
    QCDLUserError,
    Register,
    Scope,
    print_qcdl,
    procedure,
    qcdl,
)
from dwave.gate.qcdl.components import (
    IndexerMixin,
    Procedure,
    QCDLModule,
    QCDLStatementBridge,
    objwalk,
)
from dwave.gate.qcdl.qcdl_circuit import QCDLCircuit
from dwave.gate.qcdl.qcdl_models import QCDLStatement


def _check_serializable(data):
    # ensure that qcdls are pickleable
    assert pickle.loads(pickle.dumps(data)) == data
    raw = data.model_dump(exclude_unset=True)
    assert json.loads(json.dumps(raw)) == raw


def test_unwrap():
    """Check that unwrap creates the same structure."""
    for cont in [
        1,
        list(range(5)),
        [list(range(5))],
        dict(a=list(range(5)), b="b"),
        dict(a=dict(a="a", b=list(range(5)))),
        [1, dict(a=1, b=2), 2],
        np.arange(10),
    ]:
        new_cont = QCDLModule.unwrap(cont)

        if not isinstance(cont, int):
            assert cont is not new_cont

        if isinstance(cont, np.ndarray):
            np.testing.assert_allclose(cont, new_cont)
        else:
            assert cont == new_cont


def test_procedure():
    @qcdl(2)
    def main(q0, q1, **kwargs):
        @procedure
        def bell(q0, q1, **kwargs):
            q0.h()
            q0.cx(q1)
            q0.measure()
            q1.measure()

        bell(q0, q1)

    val = main().model_dump(exclude_unset=True)
    print_qcdl(val)

    assert set(val["procedures"].keys()) == {"bell_q0_q1"}


def test_nested_procedure_definitions():
    @qcdl(3)
    def h2(q0, q1, **kwargs):
        for q in [q0, q1, kwargs["q2"]]:
            assert q._proc.proc_name.startswith("main")
            assert q._proc.proc_name == "main"

        @procedure
        def ansatz(q0, q1, **kwargs):
            for q in [q0, q1, kwargs.get("q2")]:
                if q:
                    assert q._proc.proc_name.startswith("ansatz")
            q0.h()
            q0.cx(q1)

        @procedure
        def measure(*qubits):
            for q in qubits:
                assert q._proc.proc_name.startswith("measure")
                q.measure()

        @procedure
        def extra_nest(two_qubits=None):
            q0, q1 = two_qubits
            assert q0.qcdl_module_name == "q0"
            assert q1.qcdl_module_name == "q1"
            for q in two_qubits:
                assert q._proc.proc_name.startswith("extra_nest")

            ansatz(q0, q1, q2=kwargs["q2"])
            q0.u3(1, 0, 1, units="fractions")
            q1.u3("U_0,0,0")
            measure(q0, q1)

            ansatz(q1, q0)
            q0.u3("U_0,0,0")
            q1.u3(1, 0, 1, units="fractions")
            measure(q0, q1)

        extra_nest(two_qubits=[q0, q1])

    val = h2().model_dump(exclude_unset=True)
    _check_serializable(h2())
    print_qcdl(val)
    assert set(val["procedures"]) == {
        "ansatz_q0_q1_q2_q2",
        "ansatz_q1_q0",
        "measure_q0_q1",
        "extra_nest_two_qubits_0_q0_two_qubits_1_q1",
    }


def test_procedure_registers_unique():
    num_unique = 3

    @procedure
    def my_proc(reg: Register):
        reg += 1

    @qcdl(3)
    def main(q0, q1, q2):
        sc = Scope(q0, q1, q2)
        for idx in range(num_unique):
            r = sc.Register(name=f"reg{idx}")
            my_proc(r)

    qcdl_input = main().model_dump(exclude_unset=True)
    assert len(qcdl_input["procedures"]) == num_unique


def test_scope_id():
    scope_id_container = []

    @qcdl(3)
    def main(q0, q1, q2):
        sc = Scope(q0, q1, q2)
        scope_id_container.append(sc.scope_id)
        with sc.Repeat(3):
            sc.barrier()

    qcdl_input = main().model_dump(exclude_unset=True)
    scope_id = scope_id_container.pop()
    assert isinstance(scope_id, int)

    for stmt in qcdl_input["program"]["statements"]:
        s = QCDLStatement.model_validate(stmt)
        assert s.kwargs["scope_id"] == scope_id


def test_procedure_op_key():
    num_qubits = 4

    @procedure
    def my_proc(sender: QCDLModule, register: Register):
        sender.x()
        register += 1

    @qcdl(num_qubits)
    def main(**kwargs):
        qubits = list(kwargs.values())
        sc = Scope(*qubits)
        r = sc.Register(name="reg")
        for qubit in qubits:
            my_proc(sender=qubit, register=r)

    qcdl_input = main().model_dump(exclude_unset=True)
    print_qcdl(qcdl_input)
    assert len(qcdl_input["procedures"]) == num_qubits


def test_serialization_error():
    @qcdl(3)
    def f1(q0, q1, q2):
        q0.some_method1(param=123)
        q0.some_method2(param=q0.not_an_element)

    class ExampleEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, QCDLStatementBridge):
                raise QCDLUserError(f"Unresolved statement {obj}")

            try:
                return json.JSONEncoder.default(self, obj)
            except TypeError as err:
                raise QCDLUserError(f"Can't serialize {obj}") from err

    val = f1().model_dump(exclude_unset=True)

    with pytest.raises(QCDLUserError, match="not_an_element"):
        for path, a in objwalk(val):
            if isinstance(a, QCDLStatementBridge):
                dotted_path = ".".join([str(p) for p in path])
                raise QCDLUserError(f"Unresolved statement {a} at {dotted_path}")

    with pytest.raises(QCDLUserError, match="not_an_element"):
        print(json.dumps(val, indent=4, cls=ExampleEncoder))


def test_unique_statements_to_procs():
    """Check two different procedures can not be created with same name."""

    @qcdl(2)
    def main(q0, q1, **kwargs):
        container = dict(my_variable=123)

        @procedure
        def bell(q0, q1, **kwargs):
            q0.h()
            q0.eswap(q1, theta=container["my_variable"])
            q0.measure()
            q1.measure()

        bell(q0, q1)
        container["my_variable"] += 1
        bell(q0, q1)

    with pytest.raises(QCDLUserError, match="can not overwrite procedure"):
        main()


def test_get_next_index():
    circuit = QCDLCircuit()
    objs = [circuit, IndexerMixin(), Procedure("myproc", circuit)]

    for obj in objs:
        seen = set()
        for _ in range(3):
            idx = obj.get_next_index("test")
            assert idx not in seen
            seen.add(idx)

            idx2 = obj.get_next_index("test2")
            assert idx2 in seen
