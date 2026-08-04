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

"""Utilities for handling results returned by the simulator or QPU."""

from __future__ import annotations

import base64
import datetime
import enum
import functools
import io
import logging
from collections import Counter
from typing import Any, Iterable, cast

import numpy as np
import polars as pl

from dwave.gate.qcdl import LogicalOutcomeToInteger
from dwave.gate.qcdl.records import RecordFormat
from dwave.gate.result_schema import ResultDict


logger = logging.getLogger(__name__)

RegisterType = list[str | None]

DEFAULT_TAG = ""


def get_default_register(qubits: Iterable[str]) -> RegisterType:
    """The register used if none is provided.

    This register will include all qubits in the circuit, regardless of which
    have measurements. This is usually not what is desired.

    Sorts qubits to have the highest index first, e.g., ``[q2, q1, q0]``

    Args:
        qubits: list of qubits in system

    Returns:
        Qubits sorted highest index to lowest.
    """
    return cast(
        RegisterType, sorted(qubits, key=lambda qubit: int(qubit[1:]), reverse=True)
    )

def _encode_measurement_array(measurements: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, arr=measurements)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def _decode_measurement_array(measurements: str, to_bit: bool = True) -> np.ndarray:
    decoded = base64.b64decode(measurements)
    meas = np.load(io.BytesIO(decoded))["arr"]
    if not to_bit:
        return meas

    splat_value = LogicalOutcomeToInteger.SPLAT.value
    if np.any(meas == splat_value):
        meas = np.array(meas, dtype=str)
        meas[meas == str(splat_value)] = LogicalOutcomeToInteger.SPLAT.as_bit
    elif np.any(meas == 2):
        # backwards compatibility
        meas = np.array(meas, dtype=str)
        meas[meas == "2"] = LogicalOutcomeToInteger.SPLAT.as_bit
    else:
        meas = np.array(meas, dtype=str)
    return meas


def format_memory(
    measurements: list[Any] | dict[str, list[int | str] | np.ndarray],
    shots: int,
    register: RegisterType | None = None,
    unmeasured_value: int | str = "_",
) -> np.ndarray:
    """Shape results

    The measurement input is simply a flat array of all the measurements on each
    qubit over the entire calculation (i.e., for all shots). This data structure
    itself is naive to how many measurements occurred per shot. If a tag was
    provided to the `measure` instructions, then this method would be called
    once for each tag.

    The primary purpose of this method is to handle the possibility that each
    qubit may have been measured multiple times per shot and that the number can
    be different between qubits. This method will add padding to the
    measurements if the number of measurements is different so that we can make
    a dict of bitstrings from the data.

    NOTE: after converting the individual measurement arrays to a numpy array of
    type str, it will update the input measurements data structure in-place.

    Args:
        measurements: The measurements 2D array or dict of arrays from either
            the simulator or qpu. If a 2D array is passed the qubit name is
            inferred from the index in the array.
        shots: The number of shots to produce this data.
        register: List of qubit names (or None) to go into the register. A None in 
            the list will fill in the unmeasured_value. If a register is not
            provided, get_default_register will be used to create one.
        unmeasured_value: What to put in the register if a requested qubit wasn't measured.
            Defaults to "_".

    Raises:
        ValueError: register mismatch with results

    Returns:
        A 3D array of str with shape (measurements per shot, shots, qubits)
    """
    if isinstance(measurements, dict):
        qubit_name_to_loc: dict[str, Any] = {qn: qn for qn in measurements}
    else:
        qubit_name_to_loc = {f"q{i}": i for i in range(len(measurements))}

    if not qubit_name_to_loc:
        raise ValueError("the measurements instance has no qubits")

    if not shots:
        raise ValueError("shots is required")

    measurements_per_shot: dict[str, int] = {}

    for qn, loc in qubit_name_to_loc.items():
        meas = measurements[loc]
        if not (isinstance(meas, np.ndarray) and meas.dtype.type == np.str_):
            meas = np.array(meas, dtype=str)
            measurements[loc] = meas
        num_meas = len(meas)
        mps = num_meas / float(shots)
        if num_meas % shots == 0:
            measurements_per_shot[qn] = int(mps)
        else:
            raise ValueError(
                f"inconsistent number of measurements per shot {mps} for shots = "
                f"{shots} and num measurements = {num_meas} for qubit {qn}"
            )

    max_mps: int = max(measurements_per_shot.values())

    if register is None:
        register = get_default_register(qubit_name_to_loc.keys())

    allowed_register_values: set[Any] = set(qubit_name_to_loc.keys())
    allowed_register_values.add(None)
    if not set(register).issubset(allowed_register_values):
        raise ValueError(
            f"register {register} is not subset of"
            f" allowed qubits {allowed_register_values}"
        )

    reg_no_nones = [q for q in register if q is not None]
    if len(set(reg_no_nones)) != len(reg_no_nones):
        raise ValueError(f"repeated qubits in register {register} is not supported")

    memory_list = []
    for qubit_name in register:
        if qubit_name is None:
            mps = 1
            # this is to support qiskit -- a qubit which isn't measured seems to
            # result in a zero in the bitstring
            measurement_array = np.array([unmeasured_value] * shots).reshape(
                (shots, mps)
            )

        else:
            loc = qubit_name_to_loc[qubit_name]
            mps = measurements_per_shot[qubit_name]
            qubit_measurements = cast(np.ndarray, measurements[loc])

            # all the measurements this qubit made, shaped by its mps
            measurement_array = qubit_measurements.reshape((shots, mps))

        if max_mps > mps:
            # only do padding if size does not match. pad up to the max_mps
            measurement_array = np.pad(
                measurement_array,
                [(0, 0), (0, max_mps - mps)],
                "constant",
                constant_values=unmeasured_value,
            )
        memory_list.append(measurement_array.reshape((shots, max_mps)))
    memory = np.array(memory_list)

    memory = np.transpose(memory)
    if memory.shape != (max_mps, shots, len(register)):
        raise RuntimeError(
            f"unexpected memory shape {memory.shape!r}, "
            f"expected {(max_mps, shots, len(register))!r}"
        )

    return memory


def count_measurements(
    memory: list | np.ndarray,
    key_format: str | None = "bin",
    post_select: bool = True,
) -> dict[int | str, int]:
    """Convert memory to a dict of observation counts.

    Raw data (memory) is just 0, 1, or ``"*"`` (called "splat") for each qubit,
    for each shot. This method will convert that into a count of each set of
    measurements. The key for the dict is customizable.

    When converting to int/bin, this treats the qubit register as big endian,
    i.e., that the register has the least significant bit last in the array for
    each shot.

    :data:`key_format` values:
        * ``"bin"``: keys will be binary numbers with width taken from the memory
        * ``"hex"``: keys will be hexadecimal strings (e.g., ``'0x3'``)
        * ``None``: keys will be integers

    Examples:
        >>> count_measurements([[0,1,0,0]]*10, key_format='bin')
        {'0100': 10}
        >>> count_measurements([[0,1,0,0]]*10, key_format='hex')
        {'0x4': 10}
        >>> count_measurements([[0,1,0,0]]*10, key_format=None)
        {4: 10}

    Args:
        memory: a shots x qubits 2D array, sorted least significant
            bit is last / on the right.
        key_format: how to format the ints which will be used
            as keys. This is ignored if the memory consists of strings.
        post_select: If True, any bitstrings that have ``"*"`` in them
            will be excluded.

    Returns:
        A dict counting occurrences of each formatted shots value

    """
    if isinstance(memory, list):
        memory = np.array(memory)

    # normalization transformations
    if len(memory.shape) == 1:
        # assume it's a single shot
        memory = np.array([memory])
    elif len(memory.shape) == 3 and len(memory) == 1:
        # measurements_per_shot == 1
        memory = memory[0]

    if not isinstance(memory, np.ndarray):
        raise TypeError(
            f"memory instance needs to be a numpy array, but have {type(memory)}"
        )

    if len(memory.shape) != 2:
        raise ValueError(
            f"memory needs to be a 2D array of shots X qubits,"
            f" but shape is {memory.shape}"
        )

    # [[1 0]
    #  [0 0]
    #  [1 1]
    #  ...
    #  [1 0]
    #  [0 1]
    #  [0 0]]

    if post_select and memory.dtype.type == np.str_:
        splat = "*"
        mask = ~(memory == splat).any(axis=1)
        memory = cast(np.ndarray, memory[mask])

    # convert str array to integer array
    if (memory.dtype.type == np.str_) and (not np.all(np.char.isdigit(memory))):
        # if the input was not digits, then don't format as if it is integers
        return dict(Counter(map("".join, memory)))
    else:
        # in case the dtype was float somehow
        memory = np.array(memory, dtype=int)

    keys = memory.dot(1 << np.arange(memory.shape[-1] - 1, -1, -1))
    # [2 0 3 ... 2 1 0]

    if key_format == "bin":
        key_format = "{0:>0" + str(memory.shape[1]) + "b}"

    if key_format == "hex":
        # convert integer to hex string
        keys = np.char.mod("%#x", keys.flatten())
        # ['0x2' '0x0' '0x3' ... '0x2' '0x1' '0x0']
    elif key_format is not None:
        keys = [key_format.format(v) for v in keys]

    # make histogram
    histogram: dict[int | str, int] = dict(Counter(keys))
    # if key_format is "hex", then for this example:
    # {'0x2': 403, '0x0': 3264, '0x3': 134, '0x1': 1199}

    return histogram


class Result:
    """A container and API for a result.

    Args:
        result: This is the dict response from the simulator or jcs
            server.
        job_dir: If the job used a local directory
            to store data, it it is indicated here. Defaults to None.
    """

    def __init__(
        self,
        result: dict,
    ):
        ResultDict.model_validate(result)
        self._result: dict = result

    @property
    def result(self) -> dict:
        """The unprocessed result object."""
        return self._result

    @functools.cached_property
    def start_time(self) -> datetime.datetime | None:
        """The time the simulation or QPU run started.

        Returns:
            Start time if available.
        """
        if start_time := self._result.get("start_time"):
            return datetime.datetime.fromisoformat(start_time).replace(
                tzinfo=datetime.timezone.utc
            )
        else:
            return None

    @functools.cached_property
    def end_time(self) -> datetime.datetime | None:
        """The time the simulation or QPU run ended.

        Returns:
            End time if available.
        """
        if end_time := self._result.get("end_time"):
            return datetime.datetime.fromisoformat(end_time).replace(
                tzinfo=datetime.timezone.utc
            )
        else:
            return None

    @functools.cached_property
    def run_time(self) -> float | None:
        """The time the simulation or QPU ran.

        Returns:
            Run time in seconds, if available.
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        else:
            return None

    @property
    def shots(self) -> int:
        """Number of shots in this result."""
        return self.result["shots"]

    @property
    def tags(self) -> tuple[str, ...]:
        """The tags available in the measurements.

        NOTE: tags is not currently supported with real-time-measurements.

        Returns:
            The tags.
        """
        measurements = self.get_measurements()
        if not measurements:
            return tuple()
        elif isinstance(measurements, dict):
            return tuple(measurements.keys())

        return (DEFAULT_TAG,)

    @property
    def default_tag(self) -> str:
        """The default tag, if defined.

        If there is only one tag, then this method will return it. This is
        sufficient for many experiments which only have one measurement per
        qubit.

        If there are multiple tags present in the data or no tag, then there is
        no "default" tag. In either of these cases this method will raise
        ValueError.

        Returns:
            The default tag.
        """
        if len(self.tags) == 1:
            return self.tags[0]
        elif len(self.tags) == 0:
            raise ValueError("this result have no tags")
        else:
            raise ValueError(
                "this result has multiple tags and thus doesn't have a default"
            )

    def get_measurements_register(
        self, tag: str | None = None, descending: bool = True
    ) -> list[str]:
        """Get the register inferred from the measurements data for a particular
        tag.

        This method will include a qubit in the register if and only if it has
        measurements in the log data. It determines the name of the qubit from
        its index in the measurements array.

        Args:
            tag: Which data set to load. Defaults to
                :attr:`dwave.gate.results.Result.default_tag`.
            descending: Whether qubits are in descending order. Defaults to True.

        Returns:
            A list of qubit names.
        """
        if tag is None:
            tag = self.default_tag
        measurements = self.get_measurements()
        if not measurements:
            return []
        register = [
            f"q{q}" for q, meas in enumerate(measurements[tag]) if len(meas) > 0
        ]
        if descending:
            register = list(reversed(register))
        return register

    def get_memory(
        self,
        tag: str | None = None,
        register: RegisterType | None = None,
        unmeasured_value: int | str = "_",
        shots: int | None = None,
    ) -> np.ndarray:
        """Memory is the 3D array of measurements per shot x shots x qubits

        These are the raw bits returned from a statement like ``q0.measure()``.

        Args:
            tag: Which data set to load. Defaults to
                :attr:`dwave.gate.results.Result.default_tag`.
            register: List of qubit names to include in the register; determines
                the inner dimension of the returned value.
            unmeasured_value: What to put in the register if a requested qubit
                wasn't measured.
            shots: Overrides the shots in the result object.

        Returns:
            Memory array.
        """
        if tag is None:
            tag = self.default_tag
        measurements = self.get_measurements()
        if measurements is None:
            raise RuntimeError("measurements are not available for this result")
        return format_memory(
            measurements[tag],
            register=register,
            unmeasured_value=unmeasured_value,
            shots=shots or self.shots,
        )

    def get_counts(
        self,
        tag: str | None = None,
        register: RegisterType | None = None,
        post_select: bool = False,
        unmeasured_value: int | str = "_",
        shots: int | None = None,
    ) -> list[dict[int | str, int]]:
        """A counts dict is a summation over all the shots.

        This returns a list of dicts, one for each "measurement per shot".

        Args:
            tag: Which data set to load. Defaults to
                :attr:`dwave.gate.results.Result.default_tag`.
            register: Forwarded to get_memory.
            post_select: If the counts dict should include splats or not.
            unmeasured_value: What to put in the register if a requested qubit
                wasn't measured.
            shots: Overrides the shots in the result object.

        Returns:
            A summation of the memory.
        """
        if tag is None:
            tag = self.default_tag
        memory = self.get_memory(
            tag=tag,
            register=register,
            unmeasured_value=unmeasured_value,
            shots=shots,
        )
        return [
            count_measurements(mem, key_format="bin", post_select=post_select)
            for mem in memory
        ]

    def get_records(self) -> dict | None:
        """Records are the data generated by append_table_row"""
        records = self.result.get("records")
        if records is None:
            return None

        if self.result["record_format"] == RecordFormat.POLARS.value:
            for tables in records.values():
                for table_name, table_data in tables.items():
                    if isinstance(table_data, str):
                        parquet_b64_data = tables[table_name]
                        parquet_bytes = base64.b64decode(parquet_b64_data)
                        df = pl.read_parquet(parquet_bytes)
                        tables[table_name] = df

        return records

    def get_measurements(self) -> dict[str, list[np.ndarray]] | None:
        """Measurements are the data generated by log=True"""
        measurements = self.result.get("measurements")
        if measurements:
            for data in measurements.values():
                for loc, meas in enumerate(data):
                    if isinstance(meas, str):
                        data[loc] = _decode_measurement_array(meas)
                    elif not (
                        isinstance(meas, np.ndarray) and meas.dtype.type == np.str_
                    ):
                        data[loc] = np.array(meas, dtype=str)
            return measurements
        else:
            return None


class YieldHandling(enum.StrEnum):
    """If an error was detected during an end-of-line measurement,
    it is marked with a ``"*"`` (splat) instead of a 0 or 1. Qiskit can not
    accept splats in the counts dict it is given, so they must be removed. This
    is an enumeration of some techniques offered.
    """

    only_post_selected_counts = enum.auto()
    """This means that only the post-selected shots are returned (i.e., results
    with only 0 and 1). The number of shots returned may be fewer than the
    number of shots requested."""

    renormalize_distribution = enum.auto()
    """After post selecting the distribution, this option will normalize it so
    that the sum of the values equals the number of shots requested (we divide
    by the yield).

    This approach may be necessary for code which can not handle a different
    number of shots returned than what was requested. The counts become floats.

    .. WARNING::
        This approach misrepresents the statistical errors, and in the case
        of low yield, disastrously so.

    .. WARNING::
        This will raise an exception if no shots are returned.
    """

    renormalize_distribution_or_raise = enum.auto()
    """This is the same as ``renormalize_distribution``, but it will raise an
    exception if the yield is below 10%."""

    ignore_splats = enum.auto()
    """Don't alter the distribution."""

    @staticmethod
    def from_name(name: str | YieldHandling) -> YieldHandling:
        if isinstance(name, YieldHandling):
            return name
        else:
            return YieldHandling[name]

    def apply(
        self, distribution: dict[Any, int]
    ) -> tuple[dict[Any, float | int], float]:
        num_executed_shots: int = sum(distribution.values())
        post_selected = {k: v for k, v in distribution.items() if "*" not in k}
        if not post_selected:
            # Put one key in the dict at least. Use a previously existing key as
            # a template so that the formatting is correct.
            template_key: str = next(iter(distribution))
            zeros_key = template_key.replace("1", "0").replace("*", "0")
            post_selected[zeros_key] = 0

        num_post_selected_shots: int = sum(post_selected.values())
        observed_yield: float = num_post_selected_shots / num_executed_shots

        if observed_yield == 0 and self in [
            YieldHandling.renormalize_distribution,
            YieldHandling.renormalize_distribution_or_raise,
        ]:
            raise ZeroDivisionError("yield is zero, can't renormalize distribution")
        elif (
            num_post_selected_shots < 10
            and observed_yield < 0.1
            and self == YieldHandling.renormalize_distribution_or_raise
        ):
            raise ValueError(
                f"{observed_yield=} is too low to get reasonable statistics"
            )

        msg = (
            f"post selected shot count is {num_post_selected_shots}"
            f" from yield {observed_yield:.2e}"
        )
        if num_post_selected_shots > 100 or observed_yield > 0.1:
            logger.info(msg)
        elif num_post_selected_shots > 0:
            logger.warning(msg)
        else:
            logger.error("experiment yield was 0")

        if self == YieldHandling.ignore_splats:
            return cast(dict[Any, float | int], distribution), observed_yield
        elif self == YieldHandling.only_post_selected_counts:
            return cast(dict[Any, float | int], post_selected), observed_yield
        elif self in [
            YieldHandling.renormalize_distribution,
            YieldHandling.renormalize_distribution_or_raise,
        ]:
            renormalized = {
                k: float(v) / observed_yield for k, v in post_selected.items()
            }
            return renormalized, observed_yield
        else:
            raise ValueError(f"yield_handling {self} is not implemented")
