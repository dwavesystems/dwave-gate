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

import pytest

from dwave.gate.qcdl.constants import FLOAT_TO_INT, INT_TO_FLOAT
from dwave.gate.qcdl.exceptions import QCDLRuntimeError
from dwave.gate.qcdl.records import RecordFormat, RecordOutput, RecordOutputToken


def test_RecordOutput_handling():
    """This test does some shape round trips. This helps confirm
    that we can convert the shape into a key."""
    test_v = [
        "1,2,3",
        "1,(2,3)",
        "1,(2,[3])",
        "True, 2, False",
        "1,(2,1),[3]",
        "'Double', 2, 3",
        "[(5,'Double', ('Boolean', 'my-literal'))], 4",
    ]

    for v in test_v:
        for outer in [None, "tuple", "array"]:
            if outer == "tuple":
                wrapped_v = "({v},)".format(v=v)
            elif outer == "array":
                wrapped_v = "[{v}]".format(v=v)
            else:
                wrapped_v = v

            ro = RecordOutput.from_string(wrapped_v)
            assert ro.balanced
            desc = ro.description.replace(" ", "")

            # prepare the variant we can compare with
            comp_v = wrapped_v.replace(" ", "")

            if outer is None and (comp_v[0] not in "[(" or comp_v[-1] not in "[("):
                # an implicit tuple is added
                comp_v = "({comp_v})".format(comp_v=comp_v)
            elif outer == "tuple" and comp_v.endswith(",)"):
                # remove the forced tuple
                comp_v = comp_v[:-2] + ")"

            # this asserts we can round-trip
            assert desc == comp_v

            for meth in [
                "values_as_log",
                "values_as_key",
                "values_as_dict",
                "values_as_list",
            ]:
                # just check nothing crashes
                assert getattr(ro, meth)(range(ro.num_primitives))


def test_RecordOutput_empty():
    """test for an empty RecordOutput"""
    ro = RecordOutput()
    assert ro.description == "<empty shape>"
    assert ro.balanced

    vlog = ro.values_as_log([], record_type="EMPTY", ensure_one=True)
    assert vlog == [{"record_type": "EMPTY"}]
    assert ro.values_as_key([]) is None


def test_RecordOutput():
    """This test looks at all the output for a specific shape"""
    ro = RecordOutput()
    ro.array_start()
    ro.tuple_start()
    assert not ro.balanced

    sval = "my literal"
    name_of_literal = "literal's name"
    ro.literal(sval, name=name_of_literal)
    name_of_res = "my_res"
    ro.result(name_of_res)
    ro.boolean()

    ro.tuple_end()
    ro.array_end()
    assert ro.balanced

    ro.array_start()
    ro.integer()
    ro.double()
    ro.boolean()
    ro.array_end()

    assert ro.balanced
    assert ro == ro
    assert RecordOutput.from_list(ro.to_list()) == ro
    assert ro != RecordOutput()
    assert ro.num_primitives == 5

    num_tokens = len(ro._tokens)
    assert len(list(ro.data())) == num_tokens
    assert len(ro._names) == num_tokens

    assert (
        ro.description
        == "([('my literal', 'Result', 'Boolean')], ['Integer', 'Double', 'Boolean'])"
    )

    ival = 123
    fval = INT_TO_FLOAT * 4321

    for rval in [0, 1]:
        for bval in [True, False]:
            values = [rval, bval, ival, fval * FLOAT_TO_INT, bval]
            assert ro.values_as_key(values) == (
                [(sval, rval, bval)],
                [ival, fval, bval],
            )
            assert ro.values_as_list(values) == [
                sval,
                rval,
                bval,
                ival,
                fval,
                bval,
            ]
            assert ro.values_as_dict(values) == {
                name_of_literal: sval,
                "my_res": rval,
                "value1": bval,
                "value2": ival,
                "value3": fval,
                "value4": bval,
            }

            vlog = ro.values_as_log(values, record_format=RecordFormat.QIR_V2_1)

            assert vlog == [
                {"name": None, "record_type": "OUTPUT", "type": "ARRAY", "value": 1},
                {"name": None, "record_type": "OUTPUT", "type": "TUPLE", "value": 2},
                {"record_type": name_of_literal, "value": sval},
                {
                    "name": name_of_res,
                    "record_type": "OUTPUT",
                    "type": "RESULT",
                    "value": rval,
                },
                {"record_type": "OUTPUT", "type": "BOOL", "value": bval},
                {"name": None, "record_type": "OUTPUT", "type": "ARRAY", "value": 3},
                {"record_type": "OUTPUT", "type": "INT", "value": ival},
                {"record_type": "OUTPUT", "type": "DOUBLE", "value": fval},
                {"record_type": "OUTPUT", "type": "BOOL", "value": bval},
            ]


def test_errors_basic():
    error_message = "my dummy error message"
    ro = RecordOutput()
    ro.literal("my message")

    # normally the check_error and error calls happen in completely
    # different contexts. interleaving the calls like this only makes sense
    # in a unit test
    ro.check_error(raise_if_error=True)
    ro.error(error_message)
    with pytest.raises(QCDLRuntimeError) as e:
        ro.check_error(raise_if_error=True)

    assert ro.balanced
    assert ro == ro
    assert RecordOutput.from_list(ro.to_list()) == ro

    assert error_message in str(e.value)
    ro.check_error(raise_if_error=False)


def test_record_format():
    assert RecordFormat.from_record_formatter("arrays") == RecordFormat.ARRAYS
    assert RecordFormat.from_record_formatter(None) == RecordFormat.RAW
    assert RecordFormat.from_record_formatter("log").is_log_format


def test_format_v21():
    metadata = [
        "no_value",
        ("name1", "value1"),
        ("name2", "value2"),
    ]

    ro = RecordOutput()

    for meta in metadata:
        if isinstance(meta, tuple):
            name, value = meta
            ro.literal(
                "{name}\t{value}".format(name=name, value=value), name="METADATA"
            )
        else:
            ro.literal(meta, name="METADATA")

    ro.array_start(name="my_array")

    ro.tuple_start(name="my_tuple")
    name_of_res = "my_result"
    ro.result(name_of_res)
    ro.boolean(name="my_bool")
    ro.tuple_end()
    ro.array_end()

    ro.array_start(name="my_array")
    ro.integer(name="my_int")
    ro.double(name="my_double")
    ro.boolean(name="my_bool")
    ro.array_end()

    assert ro.balanced
    assert ro == ro
    assert RecordOutput.from_list(ro.to_list()) == ro

    log_data = ro.values_as_log(
        [0] * ro.num_primitives, record_format=RecordFormat.QIR_V2_1
    )
    print(json.dumps(log_data, indent=4))

    assert len(log_data) > ro.num_primitives + len(metadata)
    container_lengths = [1, 2, 3]
    for item in log_data:
        assert "time_stamp" not in item

        if item["record_type"] == "METADATA":
            assert isinstance(item["value"], str)
        else:
            assert item["record_type"] == "OUTPUT"
            value = item["value"]
            assert item["name"] == "my_" + item["type"].lower()

            if item["type"] in ["ARRAY", "TUPLE"]:
                assert value == container_lengths.pop(0)
            else:
                assert isinstance(value, (int, float))
                assert value == 0
                assert item["type"] in ["RESULT", "INT", "DOUBLE", "BOOL"]
    assert not container_lengths


def test_format_v21_end():
    shape = RecordOutput()
    exit_code = 321
    shape.literal(exit_code, name="END")

    log_data = shape.values_as_log(
        [], record_type="END123", ensure_one=True, record_format=RecordFormat.QIR_V2_1
    )
    assert len(log_data) == 1
    item = log_data[0]
    assert item["record_type"] == "END"
    assert item["value"] == exit_code
    assert len(item) == 2


def test_format_v1_end():
    shape = RecordOutput()
    log_data = shape.values_as_log(
        [], record_type="END", ensure_one=True, record_format=RecordFormat.QIR_V2_1
    )
    assert len(log_data) == 1
    item = log_data[0]
    assert len(item) == 1
    assert item["record_type"] == "END"


@pytest.mark.parametrize("post_select", [0, 1])
def test_post_select(post_select):
    shape = RecordOutput()
    shape.integer("post_select")

    log_data = shape.values_as_log(
        [post_select], record_type="END", record_format=RecordFormat.QIR_V2_1
    )
    if post_select == 1:
        assert log_data is None
    else:
        assert len(log_data) == 1
        item = log_data[0]
        assert len(item) == 1
        assert item["record_type"] == "END"


def test_exit():
    for val in [0, "zero"]:
        ro = RecordOutput()
        ro.integer()
        ro.double()
        ro.exit(val)
        assert ro.error_message is None
        assert ro.exit_code == (RecordOutputToken.EXIT, val)
        assert isinstance(ro.to_list(), list)
        assert len(ro.to_list()) == 3
        assert ro.num_primitives == 2

    for val in [0, "zero"]:
        ro = RecordOutput()
        ro.integer()
        ro.error(val)
        assert ro.exit_code == (RecordOutputToken.ERROR, val)
        assert isinstance(ro.error_message, str)
        assert isinstance(ro.to_list(), list)
        assert len(ro.to_list()) == 2
        assert ro.num_primitives == 1
        assert str(val) in ro.error_message
        with pytest.raises(QCDLRuntimeError, match=str(val)):
            ro.check_error(raise_if_error=True)

    for val in [0, "zero"]:
        ro = RecordOutput()
        ro.integer()
        ro.literal(val)
        assert ro.error_message is None
        assert ro.exit_code is None
        assert isinstance(ro.to_list(), list)
        assert len(ro.to_list()) == 2
        assert ro.num_primitives == 1
