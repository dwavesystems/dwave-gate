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

from dwave.gate.qcdl import qcdl
from dwave.gate.qcdl.transformer import transform_qcdl


def _check_serializable(data):
    # ensure that qcdls are pickleable
    for serializer in [pickle, json]:
        serialized = serializer.dumps(data)
        unserialized = serializer.loads(serialized)
        assert data == unserialized


def test_statement_order_and_signature():
    @qcdl(3)
    def f1(q0, q1, q2):
        assert q0.qcdl_module_name == "q0"
        assert q1.qcdl_module_name == "q1"
        assert q2.qcdl_module_name == "q2"
        q2.idle(5)
        q1.idle(5)
        q0.idle(5)

    val = f1().model_dump(exclude_unset=True)
    transform_qcdl(val)
    statements = val["program"]["statements"]
    assert len(statements) == 3
    for i, stmt in enumerate(statements):
        assert stmt["qubit"] == "q%i" % (2 - i)
    assert set(val["program"]["signature"]["qubits"]) == set(
        ["q%i" % i for i in range(3)]
    )


def test_single_qubit_idle_repeats_to_three_statements():
    @qcdl(1)
    def f2(q0, **kwargs):
        for _ in range(3):
            q0.idle(5)

    val = f2().model_dump(exclude_unset=True)
    transform_qcdl(val)
    assert len(val["program"]["statements"]) == 3


def test_bell_cx_statement_contains_target_arg():
    @qcdl(2)
    def bell(q0, q1, **kwargs):
        q0.h()
        q0.cx(q1)
        q0.measure()
        q1.measure()

    val = bell().model_dump(exclude_unset=True)
    transform_qcdl(val)
    statements = val["program"]["statements"]

    assert len(statements) == 4
    assert statements[1]["args"] == ["q1"]


def test_jsonifiable():
    @qcdl(2)
    def main(q0, q1, **kwargs):
        q0.my_method(my_numpy_array=(np.zeros(10), np.ones(10)))

    val = main().model_dump(exclude_unset=True)
    _check_serializable(val)

    transform_qcdl(val)
    statements = val["program"]["statements"]
    assert statements[0]["kwargs"]["my_numpy_array"] == [[0.0] * 10, [1.0] * 10]


def test_transform_qcdl_non_dict():
    with pytest.raises(TypeError, match="QCDL must be a dictionary"):
        transform_qcdl(["not", "a", "dict"])
