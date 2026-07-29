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

"""dsl tests"""

import operator

import asteval
import numpy as np
import pytest

from dwave.gate.qcdl import (
    FixedPointRegister,
    QcdlModule,
    QCDLUserError,
    Register,
    arbitrary_function,
    procedure,
    qcdl,
)
from dwave.gate.qcdl.constants import (
    FLOAT_TO_INT,
    INT_TO_FLOAT,
    MAX_FLOAT_REGISTER_VALUE,
    MAX_INT_REGISTER_VALUE,
    MIN_FLOAT_REGISTER_VALUE,
    MIN_INT_REGISTER_VALUE,
    UNSIGNED_MAX_INT_REGISTER_VALUE,
)
from dwave.gate.qcdl.registers import (
    SENS,
    Array,
    ExpressionAggregator,
    Output,
    RegisterExpression,
    validate_memory_allocation,
    validate_name,
    validate_value,
)


def check_aeval_error(aeval):
    # asteval doesn't currently raise exceptions
    if aeval.error:
        errmsg = "\n".join(aeval.error[0].get_error())
        try:
            exc = aeval.error[0].exc
        except Exception:
            exc = RuntimeError
        raise exc(errmsg)


def _test_op(op, use_ints):
    if use_ints:
        vals = [-5, -1, 0, 1, 5, SENS]
    else:
        vals = [-1.99, -1, 0, 1.5]
    expected_results = []

    MOCK_SENS = 123

    @qcdl(3)
    def main(q0, q1, q2):
        modules = [q0]

        cls = Register if use_ints else FixedPointRegister
        a = cls(modules, name="a")
        b = cls(modules, name=None)
        c = cls(modules, name="c")

        for lhs_val in vals:
            for rhs_val in vals:
                for lhs in [a, lhs_val]:
                    for rhs in [b, rhs_val]:
                        if op == "__rshift__":
                            if rhs is b or rhs_val <= 0:
                                continue

                        expected = getattr(operator, op)(lhs_val, rhs_val)
                        expected_results.append(
                            dict(
                                assignments={a.value: lhs_val, b.value: rhs_val},
                                expected=expected,
                            )
                        )
                        c <<= getattr(operator, op)(lhs, rhs)

                        # compound statement
                        expected_results.append(
                            dict(
                                assignments={a.value: lhs_val, b.value: rhs_val},
                                expected=expected - 3 * expected,
                            )
                        )
                        c <<= getattr(operator, op)(lhs, rhs) - 3 * getattr(
                            operator, op
                        )(lhs, rhs)

    val = main().model_dump(exclude_unset=True)

    expressions = []
    for s in val["program"]["statements"]:
        if s["op"] == "cpu":
            expr = s["args"][0]
            expressions.append(expr)

    assert len(expressions) == len(expected_results)
    for raw_expr, expected in zip(expressions, expected_results):
        symtable = asteval.make_symbol_table(**expected["assignments"])
        symtable["SENS"] = MOCK_SENS
        aeval = asteval.Interpreter(symtable=symtable)
        expr = raw_expr[len("c <<=") :].strip()
        calculated = aeval.eval(expr)
        check_aeval_error(aeval)

        expected_val = expected["expected"]

        # if it had SENS, these will still be expressions
        if isinstance(calculated, RegisterExpression):
            calculated = aeval.eval(calculated.value)
        if isinstance(expected_val, RegisterExpression):
            expected_val = aeval.eval(expected_val.value)
        if calculated != expected_val:
            print(expr, expected, expected_val, calculated)
        assert isinstance(expected_val, (int, float))
        assert isinstance(calculated, (int, float))
        assert calculated == pytest.approx(
            expected_val
        )  # this is not exactly the same as assertAlmostEqual(first, second, places=7
        assert abs(calculated) < MAX_INT_REGISTER_VALUE


def test_ops():
    bin_ops = [
        "__add__",
        "__sub__",
        "__mul__",
    ]
    bitwise_ops = [
        "__and__",
        "__or__",
        "__xor__",
        "__rshift__",
    ]
    comp_ops = ["__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__"]

    for op in bin_ops + comp_ops + bitwise_ops:
        _test_op(op, use_ints=True)

    for op in bin_ops + comp_ops:
        _test_op(op, use_ints=False)


def test_compatible():
    @qcdl(3)
    def main(q0, q1, q2):
        with pytest.raises(QCDLUserError):
            Register._assert_compatible_modules(Register([q0]), Register([q1]))

        with pytest.raises(QCDLUserError):
            Register._assert_compatible_modules(Output([q0], "DYN1"), Register([q1]))

        with pytest.raises(TypeError):
            Register._assert_compatible_modules("a string", Register([q1]))

        Register._assert_compatible_modules(Register([q1]), Register([q1]))
        Register._assert_compatible_modules(Output([q1], "DYN1"), Register([q1]))

    main()


def test_validation():
    for bad_name in ["lambda", "DYN1", "a b c", "123", 123, "DYN1", "q0", "c12"]:
        with pytest.raises(QCDLUserError):
            validate_memory_allocation(bad_name, 0, 1, "int")
        with pytest.raises(QCDLUserError):
            validate_name(bad_name)


def test_int_validation():
    for bad_value in [
        MIN_INT_REGISTER_VALUE - 1,
        True,
        1.234,
        None,
        2**17,
        -(2**17) - 1,
        1j,
        MAX_INT_REGISTER_VALUE + 1,
    ]:
        with pytest.raises(QCDLUserError):
            validate_value(bad_value, "int")
        with pytest.raises(QCDLUserError):
            validate_memory_allocation("myreg", bad_value, 1, "int")

    for ok_value in [MIN_INT_REGISTER_VALUE, -5, 0, 5, MAX_INT_REGISTER_VALUE]:
        # pass values as int, list, or ndarray
        for obj in [ok_value, [ok_value] * 5, np.ones(5) * ok_value]:
            for dtype in ["int", int]:
                validate_value(obj, dtype)

    for ok_unsigned_value in [0, 5, UNSIGNED_MAX_INT_REGISTER_VALUE]:
        validate_value(ok_unsigned_value, "int", signed=False)

    for bad_unsigned_value in [-1, UNSIGNED_MAX_INT_REGISTER_VALUE + 1]:
        with pytest.raises(QCDLUserError):
            validate_value(bad_unsigned_value, "int", signed=False)


def test_float_validation():
    bad_floats = [
        True,
        None,
        MIN_FLOAT_REGISTER_VALUE - INT_TO_FLOAT,
        MAX_FLOAT_REGISTER_VALUE + INT_TO_FLOAT,
    ]
    for bad_value in bad_floats:
        with pytest.raises(QCDLUserError):
            validate_value(bad_value, "float")
    for ok_value in [
        MIN_FLOAT_REGISTER_VALUE,
        -1,
        0.25,
        0,
        1,
        MAX_FLOAT_REGISTER_VALUE,
    ]:
        # pass values as float, list, or ndarray
        for obj in [ok_value, [ok_value] * 5, np.ones(5) * ok_value]:
            for dtype in ["float", float]:
                validate_value(obj, "float")

        # all values must be ok
        for bad_value in bad_floats:
            combined = [ok_value, bad_value]
            with pytest.raises(QCDLUserError):
                validate_value(combined, "float")

        with pytest.raises(QCDLUserError):
            validate_value(ok_value, "floaties")


def test_inheritance():
    @qcdl(3)
    def main(q0, q1, q2):
        modules = [q0]

        a = Register(modules, name="a")
        b = FixedPointRegister(modules, name="b")
        with pytest.raises(TypeError):
            b | 4
        with pytest.raises(TypeError):
            b & 4

        for orig_r in [a, b]:
            r = orig_r
            r += 5
            r -= 5
            r *= 5

            with pytest.raises(TypeError):
                r / 4
            with pytest.raises(TypeError):
                r /= 4

    main()


def test_output():
    name = "DYN1"

    @qcdl(3)
    def main(q0, q1, q2):
        modules = [q0]

        with pytest.raises(QCDLUserError):
            Output(modules, "DY")

        a = Output(modules, name)
        a <<= 5

        b = Register(modules)
        a <<= b + 123

        with pytest.raises(QCDLUserError):
            b <<= a

        with pytest.raises(QCDLUserError):
            b += a

    val = main().model_dump(exclude_unset=True)
    for s in val["program"]["statements"]:
        if s["op"] == "cpu":
            expr = s["args"][0]
            assert expr.startswith(name)


def test_array():
    aname = "a_array"
    bname = "b_register"

    num_b = []

    @qcdl(3)
    def main(q0, q1, q2):
        modules = [q0]
        a = Array(modules, 10, name=aname)
        b = Register(modules, name=bname)

        for lhs in [a, b]:
            for idx_str in ["b", "5", "5*b+3"]:
                idx = eval(idx_str)
                for rhs in ["b", "3", "a[idx]", "a[idx]+(b-2)"]:
                    count = 0

                    if lhs is a:
                        a[idx] = eval(rhs)
                        if "b" in idx_str:
                            count += 1
                        if "b" in rhs:
                            count += 1
                        if "idx" in rhs and "b" in idx_str:
                            count += 1
                        num_b.append(count)
                    elif "a" in rhs:
                        b <<= eval(rhs)
                        count = 1
                        if "b" in rhs:
                            count += 1
                        if "idx" in rhs and "b" in idx_str:
                            count += 1
                        num_b.append(count)

    val = main().model_dump(exclude_unset=True)
    expressions = []
    for s in val["program"]["statements"]:
        if s["op"] == "cpu":
            expr = s["args"][0]
            expressions.append(expr)

    for count, expr in zip(num_b, expressions):
        assert aname + "[" in expr
        assert count == expr.count(bname)


def test_aggregator():
    @qcdl(3)
    def main(q0, q1, q2):
        modules = [q0, q1, q2]
        a = Register(modules, name="a")
        b = FixedPointRegister(modules, name="b")
        c = Array(modules, 10, name="c")

        # these get aggregated into one qcdl instruction
        with ExpressionAggregator(modules):
            a <<= 5
            b <<= a
            c[5] = b

        # these will be their own instructions
        a <<= b
        b <<= a

    val = main().model_dump(exclude_unset=True)
    first_statement = True
    for s in val["program"]["statements"]:
        if s["op"] == "cpu":
            assert set(s["kwargs"]["qubits"]) == set(["q1", "q2"])
            expr = s["args"][0]

            if isinstance(expr, list):
                num_expr = len(expr)
            else:
                num_expr = len([e for e in expr.splitlines() if e])

            if first_statement:
                assert num_expr == 3
                first_statement = False
            else:
                assert num_expr == 1
    assert not first_statement


def test_af():
    @qcdl(3)
    def main(q0, q1, q2):
        modules = [q0, q1, q2]

        @arbitrary_function(modules, float, float)
        def cos(x):
            return np.cos(np.pi * x)

        @arbitrary_function(modules, float, float)
        def sin(x):
            return np.sin(np.pi * x)

        reg = Register(modules, name="reg")
        amp = 0.001
        c = amp * cos(reg + 0.1)
        s = amp * sin(3.321 * reg)

        mx00 = Output(modules, "MX00")
        mx00 <<= c
        mx01 = Output(modules, "MX01")
        mx01 <<= -s

    val = main().model_dump(exclude_unset=True)
    first_statement = True
    for s in val["program"]["statements"]:
        if s["op"] == "cpu":
            assert set(s["kwargs"]["qubits"]) == set(["q1", "q2"])
            expr = s["args"][0]

            for expected_str in ["reg", "0.001"]:
                assert expected_str in expr
            assert expr.startswith("MX0")

            if first_statement:
                assert "cos(" in expr
                first_statement = False
            else:
                assert "sin(" in expr

    assert not first_statement


def test_arbitrary_function_invalid_dtype():
    with pytest.raises(TypeError, match="in_dtype must be int or float"):
        arbitrary_function([], str, float)

    with pytest.raises(TypeError, match="out_dtype must be int or float"):
        arbitrary_function([], int, str)


def _arb_stmt(val):
    return next(
        s
        for s in val["program"]["statements"]
        if s["op"] == "allocate_arbitrary_function"
    )


def test_af_string_func_float_float():
    """String-returning func (for server-side eval) must not raise with float/float dtypes.

    Regression: wrapping with np.asarray() before multiplying by out_scale=1 (int)
    triggers a numpy 2.0 string-ufunc TypeError.
    """

    @qcdl(1)
    def main(q0):
        @arbitrary_function([q0], float, float)
        def my_func(x):
            return "sin(x)"

    val = main().model_dump(exclude_unset=True)
    arb = _arb_stmt(val)
    assert arb["args"][1] == "sin(x)"
    assert arb["kwargs"]["dtype"] == "float"


def test_af_string_func_int_float():
    """String-returning func with int/float dtypes should not raise."""

    @qcdl(1)
    def main(q0):
        @arbitrary_function([q0], int, float)
        def my_func(x):
            return "cos(x)"

    val = main().model_dump(exclude_unset=True)
    arb = _arb_stmt(val)
    assert arb["args"][1] == "cos(x)"


def test_af_dtype_int_float():
    """Numeric func with int/float dtypes applies FLOAT_TO_INT input scaling."""

    @qcdl(1)
    def main(q0):
        @arbitrary_function([q0], int, float)
        def my_func(x):
            # x arrives pre-scaled by FLOAT_TO_INT; normalise back to [-1, 1)
            return np.sin(np.pi * x / FLOAT_TO_INT)

    val = main().model_dump(exclude_unset=True)
    arb = _arb_stmt(val)
    assert isinstance(arb["args"][1], list)
    assert len(arb["args"][1]) == 512


def test_af_dtype_float_int():
    """Numeric func with float/int dtypes applies INT_TO_FLOAT output scaling."""

    @qcdl(1)
    def main(q0):
        @arbitrary_function([q0], float, int)
        def my_func(x):
            # return integer-scaled values; out_scale=INT_TO_FLOAT maps them to [-2, 2)
            return np.sin(np.pi * x) * FLOAT_TO_INT

    val = main().model_dump(exclude_unset=True)
    arb = _arb_stmt(val)
    assert isinstance(arb["args"][1], list)
    assert len(arb["args"][1]) == 512


def test_register_is_container():
    @procedure
    def inner(q, register):
        q.rx(0.123)
        # this test handles the case where a register is a QcdlModuleContainer
        # see https://github.com/quantumcircuits/aqumen_environment/issues/1330
        register += 1

    @qcdl(2)
    def main(q0, q1, **kwargs):
        modules = [q0]

        reg = Register(modules, name="reg")
        inner(q1, reg)

    val = main().model_dump(exclude_unset=True)

    for s in val["program"]["statements"]:
        assert s["op"] != "cpu"

    proc = val["procedures"]["inner_q1__Register_reg__q0"]
    count = 0
    for s in proc["statements"]:
        if s["op"] == "cpu":
            count += 1
        assert s["op"] in ["cpu", "rx"]
    assert count == 1

    qubits_used = val["program"]["signature"]["qubits_used"]
    assert qubits_used == ["q0", "q1"]


def test_find_modules():
    @qcdl(2)
    def main(q0, q1, **kwargs):
        modules = [q0]

        reg = Register(modules, name="reg")
        assert next(QcdlModule.find_modules(reg)).qcdl_module_name == "q0"
        assert next(QcdlModule.find_modules(q0)).qcdl_module_name == "q0"
        assert next(QcdlModule.find_modules(q1)).qcdl_module_name == "q1"
        assert set(
            [q.qcdl_module_name for q in QcdlModule.find_modules(reg, q1)]
        ) == set(["q0", "q1"])

    assert main()


def test_parent_proc_not_altered():
    class ModuleHider(object):
        # this is not a QcdlModuleContainer and thus
        # hides a module
        def __init__(self, register):
            self.register = register

    @procedure
    def inner(q0, hider):
        q0.rx(0.123)
        # this test handles the case where something containing modules but
        # is not a QcdlModuleContainer is passed
        # see https://github.com/quantumcircuits/aqumen_environment/issues/1330
        hider.register += 1

    @qcdl(1)
    def main(q0, **kwargs):
        modules = [q0]
        reg = Register(modules, name="reg")
        hider = ModuleHider(reg)
        inner(q0, hider)

    with pytest.raises(QCDLUserError) as cm:
        main()
    assert "before child procedure" in str(cm.value)
