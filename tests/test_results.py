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


import datetime
import logging
import random

import numpy as np
import pytest
from pydantic import ValidationError

from dwave.gate.qcdl.records import RecordFormat
from dwave.gate.qcdl import LogicalOutcomeToInteger
from dwave.gate.results import (
    Result,
    YieldHandling,
    _decode_measurement_array,
    _encode_measurement_array,
    count_measurements,
    format_memory,
    get_default_register,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal(extra: dict | None = None) -> dict:
    """Minimal valid result dict (only the required field)."""
    d = {"num_shots": 10}
    if extra:
        d.update(extra)
    return d


def _make_result(extra: dict | None = None) -> Result:
    return Result.model_validate(_minimal(extra))


# ---------------------------------------------------------------------------
# Result – required field
# ---------------------------------------------------------------------------


def test_result_dict_requires_num_shots():
    with pytest.raises(ValidationError) as exc_info:
        Result.model_validate({})
    assert any(e["loc"] == ("num_shots",) for e in exc_info.value.errors())


def test_result_dict_minimal_valid():
    r = Result.model_validate({"num_shots": 5})
    assert r.num_shots == 5


# ---------------------------------------------------------------------------
# Result – extra fields allowed
# ---------------------------------------------------------------------------


def test_result_dict_allows_extra_keys():
    r = Result.model_validate({"num_shots": 1, "future_field": "value"})
    assert r.model_extra["future_field"] == "value"


# ---------------------------------------------------------------------------
# Result – start_time / end_time validation
# ---------------------------------------------------------------------------


def test_result_dict_valid_timestamps():
    r = Result.model_validate({
            "num_shots": 1,
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T01:00:00",
        }
    )
    assert isinstance(r.start_time, datetime.datetime)
    assert isinstance(r.end_time, datetime.datetime)


def test_result_dict_invalid_start_time():
    with pytest.raises(ValidationError) as exc_info:
        Result.model_validate({"num_shots": 1, "start_time": "not-a-date"})
    assert any(e["loc"] == ("start_time",) for e in exc_info.value.errors())


def test_result_dict_invalid_end_time():
    with pytest.raises(ValidationError) as exc_info:
        Result.model_validate({"num_shots": 1, "end_time": "yesterday"})
    assert any(e["loc"] == ("end_time",) for e in exc_info.value.errors())


def test_result_dict_end_before_start_rejected():
    with pytest.raises(ValidationError, match="end_time.*must not be before"):
        Result.model_validate(
            {
                "num_shots": 1,
                "start_time": "2026-01-02T00:00:00",
                "end_time": "2026-01-01T00:00:00",
            }
        )


def test_result_dict_equal_start_end_accepted():
    ts = "2026-01-01T12:00:00"
    r = Result.model_validate({"num_shots": 1, "start_time": ts, "end_time": ts})
    assert r.start_time == r.end_time


def test_result_dict_only_start_time():
    r = Result.model_validate({"num_shots": 1, "start_time": "2026-01-01T00:00:00"})
    assert r.start_time is not None
    assert r.end_time is None


# ---------------------------------------------------------------------------
# Result – record_format coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("polars", RecordFormat.POLARS),
        ("table", RecordFormat.TABLE),
        ("qir.v2.1", RecordFormat.QIR_V2_1),
    ],
)
def test_result_dict_record_format_coercion(value, expected):
    r = Result.model_validate({"num_shots": 1, "record_format": value})
    assert r.record_format is expected


def test_result_dict_invalid_record_format():
    with pytest.raises(ValidationError):
        Result.model_validate({"num_shots": 1, "record_format": "unknown_format"})


# ---------------------------------------------------------------------------
# Result – records / record_format consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", [RecordFormat.QIR_V2_1])
def test_result_dict_log_format_accepts_str_records(fmt):
    r = Result.model_validate(
        {
            "num_shots": 1,
            "record_format": fmt.value,
            "records": "HEADER\tschema\nEND\t0",
        }
    )
    assert isinstance(r.records, str)


@pytest.mark.parametrize("fmt", [RecordFormat.POLARS, RecordFormat.TABLE])
def test_result_dict_table_format_accepts_dict_records(fmt):
    r = Result.model_validate(
        {"num_shots": 1, "record_format": fmt.value, "records": {"q0": {"t": "data"}}}
    )
    assert isinstance(r.records, dict)


@pytest.mark.parametrize("fmt", [RecordFormat.QIR_V2_1])
def test_result_dict_log_format_rejects_dict_records(fmt):
    with pytest.raises(ValidationError, match="records must be a str"):
        Result.model_validate(
            {"num_shots": 1, "record_format": fmt.value, "records": {"q0": {}}}
        )


@pytest.mark.parametrize("fmt", [RecordFormat.POLARS, RecordFormat.TABLE])
def test_result_dict_table_format_rejects_str_records(fmt):
    with pytest.raises(ValidationError, match="records must be a dict"):
        Result.model_validate(
            {"num_shots": 1, "record_format": fmt.value, "records": "raw string"}
        )


def test_result_dict_records_without_format_accepted():
    # records present but no format — no consistency check fires
    r = Result.model_validate({"num_shots": 1, "records": {"q0": {"t": "x"}}})
    assert r.records == {"q0": {"t": "x"}}


# ---------------------------------------------------------------------------
# Result – measurements
# ---------------------------------------------------------------------------


def test_result_dict_measurements_list_of_lists():
    r = Result.model_validate({"num_shots": 2, "measurements": {"": [[0, 1], [1, 0]]}})
    assert r.measurements[""] == [[0, 1], [1, 0]]


def test_result_dict_measurements_numpy_arrays():
    arr = np.array([0, 1], dtype=str)
    r = Result.model_validate({"num_shots": 2, "measurements": {"tag": [arr]}})
    np.testing.assert_array_equal(r.measurements["tag"][0], arr)


# ---------------------------------------------------------------------------
# Result – model dump contains supplied values
# ---------------------------------------------------------------------------


def test_result_model_dump_contains_input_values():
    d = _minimal({"record_format": RecordFormat.QIR_V2_1.value})
    ar = Result.model_validate(d)
    dumped = ar.model_dump(mode="python")
    assert isinstance(dumped, dict)
    assert dumped["record_format"] is RecordFormat.QIR_V2_1


def test_result_num_shots():
    assert _make_result().num_shots == 10


def test_result_start_end_time_none_when_absent():
    ar = _make_result()
    assert ar.start_time is None
    assert ar.end_time is None


def test_result_start_end_time_parsed():
    ar = _make_result(
        {"start_time": "2026-01-01T00:00:00", "end_time": "2026-01-01T01:00:00"}
    )
    assert isinstance(ar.start_time, datetime.datetime)
    assert isinstance(ar.end_time, datetime.datetime)
    assert ar.run_time == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# Utility functions, encode/decode and count measurements
# ---------------------------------------------------------------------------


def test_default_register():
    qubits = [f"q{idx}" for idx in range(5, 0, -1)]
    assert get_default_register(qubits) == qubits
    assert get_default_register(reversed(qubits)) == qubits


def test_unmeasured_qubit():
    # test that if one of the registers has no measurements that the unmeasured
    # value is used correctly and that the final result has an appropriate dtype
    register = [f"q{q}" for q in range(3)]
    memory = format_memory(
        measurements=[[1, "*", 0], [1, 1, 0], []], register=register, shots=3
    )
    assert isinstance(memory, np.ndarray)
    assert memory.dtype.kind == "U"
    assert set(memory.flatten()) == set(["0", "1", "*", "_"])
    assert count_measurements(memory) == {"00_": 1, "11_": 1}

    memory = format_memory(
        measurements=[[1, 0, 0], [1, 1, 0], []], register=register, shots=3
    )
    assert isinstance(memory, np.ndarray)
    assert memory.dtype.kind == "U"
    assert count_measurements(memory) == {"00_": 1, "01_": 1, "11_": 1}


@pytest.mark.parametrize("splats", [True, False])
@pytest.mark.parametrize("val", range(8))
def test_count_measurements(splats, val):
    # simple examples for how this works
    q3fmt = "{0:>03b}"
    b = q3fmt.format(val)
    shots = 100
    if splats:
        data = np.array(
            [
                [
                    (
                        random.choices([int(v), "*"], weights=[0.75, 0.25])[0]
                        if shot > 0
                        else int(v)
                    )
                    for v in b
                ]
                for shot in range(shots)
            ]
        )
        num_post_selected = sum("*" not in row for row in data)
        # make sure at least one shot will survive post selection
        assert num_post_selected > 0
    else:
        data = np.array([[int(v) for v in b]] * shots)
        num_post_selected = shots
    assert len(data) == shots
    assert count_measurements(data, key_format="bin") == {b: num_post_selected}
    if splats:
        counts = count_measurements(data, key_format="bin", post_select=False)
        assert sum(counts.values()) == shots
        assert counts[b] >= 1

    assert count_measurements(data, key_format="hex") == {hex(val): num_post_selected}
    assert count_measurements(data, key_format=None) == {val: num_post_selected}
    assert count_measurements(data, key_format=q3fmt) == {b: num_post_selected}


@pytest.mark.parametrize("post_select", [True, False])
def test_splats_and_unmeasured(post_select):
    shots = 100
    num_qubits = 3
    data = np.array(
        [random.choices([0, 1, "*", "_"], k=shots) for _ in range(num_qubits)]
    )
    assert isinstance(
        count_measurements(data, key_format="bin", post_select=post_select), dict
    )


def make_and_test_mock_results(mps_array, shots=10, register=None):
    measurements = []
    measurements_per_shot = {}
    qubit_index_table = {}
    for i, mps in enumerate(mps_array):
        qubit_name = f"q{i}"
        measurements_per_shot[qubit_name] = mps
        qubit_index_table[qubit_name] = i

        # each measurement value is just the index of the measurement
        measurements.append(list(range(shots * mps)))

    if register is None:
        register = list(qubit_index_table.keys())

    unmeasured_value: str = "_"
    memory = format_memory(
        measurements=measurements,
        unmeasured_value=unmeasured_value,
        register=register,
        shots=shots,
    )

    max_mps = max(mps_array)
    assert memory.shape == (max_mps, shots, len(register))

    for mps in range(max_mps):
        for shot in range(shots):
            returned_register = memory[mps][shot]
            expected_register = np.array(
                [
                    (
                        str(mps + shot * measurements_per_shot[qubit_name])
                        if mps < measurements_per_shot[qubit_name]
                        else unmeasured_value
                    )
                    for qubit_name in register
                ]
            )
            assert expected_register.tolist() == returned_register.tolist()

    return memory


@pytest.mark.parametrize(
    "register", [["q1", "q2"], ["q2", "q1"], ["q1"], ["q2"], ["q0"], None]
)
def test_format_results(register):
    make_and_test_mock_results([0, 1, 2], register=register)


def test_format_results2():
    make_and_test_mock_results([2, 2, 0])
    make_and_test_mock_results([2], register=["q0"])


def _mock_results(measured_qubits, num_qubits=3, shots=10):
    measured_qubits = list(measured_qubits)
    measurements = [
        [1] * shots if q in measured_qubits else [] for q in range(num_qubits)
    ]

    return dict(
        measurements=measurements,
        shots=shots,
    )


@pytest.mark.parametrize("nones", [[], [1], [1, 2], [0, 1, 2]])
def test_none_in_register(nones, shots=10):
    """This tests the capability to put a None in the register and have it result
    in a 0 in the bitstring for that position.
    """
    full_register = ["q2", "q1", "q0"]

    register = [None if idx in nones else q for idx, q in enumerate(full_register)]
    expected = "".join("1" if q is not None else "_" for q in register)

    results = _mock_results([0, 1, 2], shots=shots)
    memory = format_memory(**results, register=register)
    assert memory.shape == (1, shots, len(register))

    counts = count_measurements(memory)
    assert counts == {expected: shots}

def test_encode_decode_measurements():
    size = 1000
    arr = np.random.randint(-1, 1, size=size)
    encoded = _encode_measurement_array(arr)
    np.testing.assert_array_equal(arr, _decode_measurement_array(encoded, to_bit=False))
    decoded = _decode_measurement_array(encoded, to_bit=True)
    expected_decoded = [LogicalOutcomeToInteger.from_outcome(v).as_bit for v in arr]
    assert set(decoded).issubset({"*", "0", "1"})
    np.testing.assert_array_equal(decoded, expected_decoded)


# ---------------------------------------------------------------------------
# Yield handling
# ---------------------------------------------------------------------------


def test_yield_handling():
    yield_100pct = {"00": 100, "01": 50}
    yield_50pct = {"00": 100, "0*": 100}
    yield_5pct = {"00": 5, "0*": 95}
    yield_0pct = {"*": 200}

    assert YieldHandling.only_post_selected_counts.apply(yield_100pct) == (
        yield_100pct,
        1.0,
    )
    assert YieldHandling.only_post_selected_counts.apply(yield_50pct) == (
        {"00": 100},
        0.5,
    )
    assert YieldHandling.only_post_selected_counts.apply(yield_0pct) == ({"0": 0}, 0)

    for dist, expected_yield in zip(
        [yield_100pct, yield_50pct, yield_0pct], [1, 0.5, 0]
    ):
        assert YieldHandling.ignore_splats.apply(dist) == (dist, expected_yield)

    for yh in [
        YieldHandling.renormalize_distribution_or_raise,
        YieldHandling.renormalize_distribution,
    ]:
        assert yh.apply(yield_100pct) == (yield_100pct, 1)
        assert yh.apply(yield_50pct) == ({"00": 200}, 0.5)

        with pytest.raises(ZeroDivisionError):
            yh.apply(yield_0pct)

    with pytest.raises(ValueError):
        YieldHandling.renormalize_distribution_or_raise.apply(yield_5pct)
