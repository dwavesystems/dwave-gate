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

import math
from collections import Counter

import numpy as np
import pytest

from dwave.gate.qcdl import QCDLUserError, Scope, print_qcdl, procedure, qcdl
from dwave.gate.qcdl.qcdl_models import QCDLStatement
from dwave.gate.qcdl.utils import simplify_float


def test_if_conditional():
    true_goto = "my_true_goto"

    @qcdl()
    def main(q0, q1):
        sc = Scope(q0, q1)
        q0.h()
        with sc.If("condition1", true_goto=true_goto):
            q0.x()
        q0.z()

    qcdl_dict = main().model_dump(exclude_unset=True)
    op_counts = Counter([stmt["op"] for stmt in qcdl_dict["program"]["statements"]])
    assert op_counts["If"] == 1
    assert op_counts["Else"] == 1
    assert op_counts["Endif"] == 1

    for stmt in qcdl_dict["program"]["statements"]:
        if stmt["op"] == "Else":
            assert stmt["kwargs"]["true_goto"] == true_goto


def test_if_true_false_conditional():
    true_goto = "my_true_goto"
    false_goto = "my_false_goto"

    @qcdl()
    def main(q0, q1):
        sc = Scope(q0, q1)
        q0.h()
        with sc.If("condition1", true_goto=true_goto, false_goto=false_goto) as Else:
            q0.x()
        with Else():
            q0.y()
        q0.z()

    qcdl_dict = main().model_dump(exclude_unset=True)
    op_counts = Counter([stmt["op"] for stmt in qcdl_dict["program"]["statements"]])
    assert op_counts["If"] == 1
    assert op_counts["Else"] == 1
    assert op_counts["Endif"] == 1

    for stmt in qcdl_dict["program"]["statements"]:
        if stmt["op"] == "Endif":
            assert stmt["kwargs"]["false_goto"] == false_goto
        elif stmt["op"] == "Else":
            assert stmt["kwargs"]["true_goto"] == true_goto


def test_nested_conditionals():
    true_goto_outer = "my_true_goto_outer"
    true_goto_inner = "my_true_goto_inner"

    false_goto_outer = "my_false_goto_outer"

    @qcdl()
    def main(q0, q1):
        sc = Scope(q0, q1)
        q0.h()
        with sc.If(
            "condition1", true_goto=true_goto_outer, false_goto=false_goto_outer
        ) as Else:
            q0.x()
            with sc.If("condition2", true_goto=true_goto_inner):
                q0.sx()
        with Else():
            q0.y()
        q0.z()

    qcdl_dict = main().model_dump(exclude_unset=True)

    op_counts = Counter([stmt["op"] for stmt in qcdl_dict["program"]["statements"]])
    assert op_counts["If"] == 2
    assert op_counts["Else"] == 2
    assert op_counts["Endif"] == 2

    endif_gotos = [true_goto_inner, None, true_goto_outer, false_goto_outer]
    for stmt in qcdl_dict["program"]["statements"]:
        if stmt["op"] == "Else":
            assert stmt["kwargs"].get("true_goto") == endif_gotos.pop(0)
        if stmt["op"] == "Endif":
            assert stmt["kwargs"].get("false_goto") == endif_gotos.pop(0)


def test_empty_conditional():
    true_goto = "my_true_goto"
    false_goto = "my_false_goto"

    @qcdl()
    def main(q0, q1):
        sc = Scope(q0, q1)
        q0.h()
        with sc.If("condition1", true_goto=true_goto, false_goto=false_goto) as Else:
            pass
        with Else():
            pass
        q0.z()

    qcdl_dict = main().model_dump(exclude_unset=True)
    op_counts = Counter([stmt["op"] for stmt in qcdl_dict["program"]["statements"]])
    assert op_counts["If"] == 1
    assert op_counts["Else"] == 1
    assert op_counts["Endif"] == 1

    for stmt in qcdl_dict["program"]["statements"]:
        if stmt["op"] == "Endif":
            assert stmt["kwargs"]["false_goto"] == false_goto
        elif stmt["op"] == "Else":
            assert stmt["kwargs"]["true_goto"] == true_goto


def test_gate_between_if_and_else_raises():
    @qcdl()
    def main(q0):
        with q0.If(True) as Else:
            q0.x()
        q0.y()
        with Else():
            q0.x()

    with pytest.raises(QCDLUserError, match="The Else context manager"):
        main()


def test_else_used_twice_raises():
    @qcdl()
    def main(q0):
        with q0.If(True) as Else:
            q0.x()
        with Else():
            q0.x()
        with Else():
            q0.z()

    with pytest.raises(QCDLUserError, match="The Else context manager"):
        main()


def test_context_error():

    @qcdl(2)
    def main_if(q0, q1):
        with q0.If("condition1"):
            q1.x()

    @qcdl(2)
    def main_else(q0, q1):
        sc = Scope(q0)
        with sc.If("condition1") as Else:
            pass
        with Else():
            q1.x()

    @qcdl(2)
    def main_qconditional_other(q0, q1):
        with q0.If(q1):
            q1.x()

    @qcdl(2)
    def main_while(q0, q1):
        sc = Scope(q0)
        with sc.While(True):
            q1.x()

    @qcdl(2)
    def main_dowhile(q0, q1):
        sc = Scope(q0)
        with sc.DoWhile(True):
            q1.x()

    @qcdl(2)
    def main_for(q0, q1):
        sc = Scope(q0)
        r = sc.Register()
        with sc.For(r, 0, True, 0):
            q1.x()

    @qcdl(2)
    def main_nested_if(q0, q1):
        sc = Scope(q0, q1)
        with sc.If("condition1"):
            with q0.If("condition2"):
                q1.x()

    @qcdl(2)
    def main_nested_else(q0, q1):
        sc = Scope(q0, q1)
        with sc.If("condition1") as Else:
            pass
        with Else():
            with q0.If("condition2"):
                q1.x()

    # these are all expected to raise an exception
    for f in [
        main_if,
        main_else,
        main_qconditional_other,
        main_while,
        main_dowhile,
        main_for,
        main_nested_if,
        main_nested_else,
    ]:
        with pytest.raises(QCDLUserError) as e:
            f()
        assert "conditional context" in e.value.args[0]

    # these are not errors
    @qcdl(2)
    def main_qconditional_same(q0, q1):
        with q0.If(q1):
            q0.x()

        q0.y()
        q1.z()

    @qcdl()
    def main_couplers(q0, c0):
        with q0.If(None):
            c0.x()

        q0.y()
        c0.z()

    for f in [
        main_qconditional_same,
        main_couplers,
    ]:
        f()


def test_non_deterministic_rtcf():
    @qcdl(2)
    def main_while(q0, q1):
        with Scope(q0, q1).While(True):
            pass
        with Scope(q0).While(True):
            pass

    @qcdl(2)
    def main_label(q0, q1):
        Scope(q0, q1).Label("label1")
        Scope(q0).Label("label2")

    @qcdl(2)
    def main_mix(q0, q1):
        with Scope(q0, q1).DoWhile(True):
            pass
        Scope(q0).Label("label2")

    @qcdl(2)
    def main_missing_from_loop(q0, q1):
        with Scope(q0).DoWhile(True):
            pass
        q1.x()

    # these are all expected to raise an exception
    for f in [
        main_while,
        main_label,
        main_mix,
        main_missing_from_loop,
    ]:
        with pytest.raises(QCDLUserError) as e:
            f()
        assert "non-deterministic" in e.value.args[0]

    # these are not errors
    @qcdl(2)
    def main_label_same(q0, q1):
        Scope(q0, q1).Label("label1")
        Scope(q0, q1).Label("label2")

    for f in [
        main_label_same,
    ]:
        f()


@pytest.mark.parametrize("ascending", [True, False])
def test_repeat(ascending):
    num = 1234

    @qcdl(3)
    def main(q0, q1, q2):
        sc = Scope(q0, q1, q2)
        with sc.Repeat(num, ascending=ascending):
            q0.x()

    qcdl_str = print_qcdl(main(), to_Display=False)
    assert qcdl_str.count(str(num)) == 1

    if ascending:
        assert qcdl_str.count(f" < {num}") == 1
    else:
        assert qcdl_str.count(" > 0") == 1
    sign = "" if ascending else "-"
    assert qcdl_str.count(f" += {sign}1") == 1
    assert qcdl_str.count("repeat_0") == 7
    assert qcdl_str.count("axis") == 0


def test_one_to_all():
    reg_name = "reg123"

    @qcdl(3)
    def main(q0, q1, q2):
        send_register = q0.Register(name=reg_name)
        sc = Scope(q1, q2)
        q0.one_to_all(sc, send_register == 1)

    input_qcdl = main().model_dump(exclude_unset=True)
    for stmt in input_qcdl["program"]["statements"]:
        s = QCDLStatement.model_validate(stmt)
        if s.op == "one_to_all":
            assert s.qubit_name == "q0"
            assert s.kwargs["qubits"] == ["q1", "q2"]
            assert s.kwargs["send"]["value"] == f"{reg_name} == 1"
        else:
            assert s.op == "allocate_memory"
            assert s.args[0] == reg_name


def test_break_outside_loop():
    @qcdl(1)
    def main(q0):
        q0.Break()

    with pytest.raises(QCDLUserError, match="not in loop, can not break"):
        main()


def test_continue_outside_loop():
    @qcdl(1)
    def main(q0):
        q0.Continue()

    with pytest.raises(QCDLUserError, match="not in loop, can not continue"):
        main()


def test_simplify_float():
    assert simplify_float(math.pi * 2) == "2*pi"
    assert simplify_float(math.nan) == "nan"
    assert simplify_float(np.nan) == "nan"


def test_validate_reused_procedures():

    def f(qa, qb, qc):
        qa.sx()
        qb.sx()
        qc.sx()

    proc_name = "my_proc_name"
    my_method1 = procedure(f, proc_name=proc_name, validate_reused_procedures=False)
    my_method2 = procedure(f, proc_name=proc_name, validate_reused_procedures=True)

    @qcdl()
    def main1(q0, q1, q2):
        for _ in range(3):
            my_method1(q0, q1, q2)

    @qcdl()
    def main2(q0, q1, q2):
        for _ in range(3):
            my_method2(q0, q1, q2)

    qcdl_dict1 = main1().model_dump(exclude_unset=True)
    qcdl_dict2 = main2().model_dump(exclude_unset=True)

    assert qcdl_dict1 == qcdl_dict2
    assert proc_name in qcdl_dict1["procedures"]
