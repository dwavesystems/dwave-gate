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

import operator

import numpy as np
import pytest
from asteval import Interpreter

from dwave.gate.qcdl import print_qcdl, procedure, qcdl
from dwave.gate.qcdl.base import Variable, VariableExpression


def test_QcdlArguments():
    @qcdl(2)
    def main(q0, q1, **kwargs):
        @procedure
        def bell(q0, q1, **kwargs):
            q0.h()
            q0.eswap(q1, theta=Variable("theta"))
            q0.eswap(q1, theta=VariableExpression("-theta"))
            q0.measure()
            q1.measure()

        bell(q0, q1)

    val = main()
    qcdl_str = print_qcdl(val, to_Display=False)
    assert "type" not in qcdl_str
    assert qcdl_str.count("`") == 2
    assert qcdl_str.count('"') == 0


def test_variable_expression_math():
    aeval = Interpreter()
    aeval("x = 13.0")
    supported_ops = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "truediv",
        "//": "floordiv",
        "%": "mod",
    }
    operands = [
        2,
        6.33,
        7,
        12.66,
        np.float64(2.5),
        -1.11,
        VariableExpression("2*x + 0.4"),
    ]
    for left in operands:
        for right in operands:
            for op, op_func_name in supported_ops.items():
                numeric = getattr(operator, op_func_name)(left, right)
                expr_left = getattr(operator, op_func_name)(
                    VariableExpression(left), right
                )
                expr_right = getattr(operator, op_func_name)(
                    left, VariableExpression(right)
                )
                expr_both = getattr(operator, op_func_name)(
                    VariableExpression(left), VariableExpression(right)
                )
                expr_in_place = getattr(operator, "i" + op_func_name)(
                    VariableExpression(left), right
                )
                for expr in [expr_left, expr_right, expr_both, expr_in_place]:
                    assert aeval(
                        VariableExpression(numeric)._variable_expression
                    ) == pytest.approx(aeval(expr._variable_expression))

    some_expr = VariableExpression("42 + x")
    with pytest.raises(TypeError):
        some_expr + (3j)


def test_variable_expression_in_place():
    aeval = Interpreter()
    operands = [2, 6.33, 7, 12.66, np.float64(2.5), -1.11]
    for left in operands:
        for right in operands:
            add = left + right
            left_add = VariableExpression(str(left))
            left_add += right
            assert aeval(left_add._variable_expression) == add

            sub = left - right
            left_sub = VariableExpression(str(left))
            left_sub -= right
            assert aeval(left_sub._variable_expression) == sub

            mul = left * right
            left_mul = VariableExpression(str(left))
            left_mul *= right
            assert aeval(left_mul._variable_expression) == mul

            truediv = left / right
            left_tdiv = VariableExpression(str(left))
            left_tdiv /= right
            assert aeval(left_tdiv._variable_expression) == pytest.approx(truediv)

            floordiv = left // right
            left_fdiv = VariableExpression(str(left))
            left_fdiv //= right
            assert aeval(left_fdiv._variable_expression) == floordiv

            mod = left % right
            left_mod = VariableExpression(str(mod))
            left_mod %= right
            assert aeval(left_mod._variable_expression) == mod
