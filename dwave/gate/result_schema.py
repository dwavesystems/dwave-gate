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

"""Pydantic schema for the result dict consumed by :class:`~dwave.gate.results.Result`."""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dwave.gate.qcdl.qcdl_models import Qcdl
from dwave.gate.qcdl.records import RecordFormat


class ResultDict(BaseModel):
    """Schema for the result dict returned by the simulator or QPU.

    Unknown keys are permitted so that future fields added by the service do
    not cause validation failures.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    num_shots: int = Field(description="Total number of shots executed.")
    start_time: str | None = Field(
        default=None, description="ISO-8601 string for when execution began."
    )
    end_time: str | None = Field(
        default=None, description="ISO-8601 string for when execution ended."
    )

    @field_validator("start_time", "end_time", mode="after")
    @classmethod
    def _validate_iso_datetime(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                datetime.datetime.fromisoformat(v)
            except ValueError:
                raise ValueError(f"{v!r} is not a valid ISO-8601 datetime string")
        return v

    run_time: float | None = Field(
        default=None, description="Wall-clock duration of the run in seconds."
    )
    seconds_per_shot: float | None = Field(
        default=None, description="Average wall-clock time per shot in seconds."
    )
    num_qubits: int | None = Field(
        default=None, description="Number of qubits used in the circuit."
    )
    simulated_qcdl: str | None = Field(
        default=None,
        description='Label describing which QCDL was simulated (e.g. ``"input"``).',
    )
    record_format: RecordFormat | None = Field(
        default=None,
        description=(
            "Serialization format used for ``records``."
            " Must be present when ``records`` is non-null."
        ),
    )
    measurements: dict[str, list[str | list[int | str] | np.ndarray]] | None = Field(
        default=None,
        description=(
            "Raw measurement data keyed by tag. Each value is a per-qubit list whose "
            "elements are either a base64-encoded compressed array (str), a flat list "
            "of int/str measurement values, or a pre-decoded NumPy array."
        ),
    )
    records: dict[str, dict[str, Any]] | str | None = Field(
        default=None,
        description=(
            "Table data produced by ``append_table_row``. "
            "A ``dict`` keyed by qubit name then table name for table formats "
            "(e.g. ``polars``), or a raw QIR log ``str`` for log formats "
            "(``qir.v1``, ``qir.v2.1``)."
        ),
    )

    @model_validator(mode="after")
    def _check_records_type_matches_format(self) -> ResultDict:
        if self.records is None or self.record_format is None:
            return self
        if self.record_format.is_log_format and not isinstance(self.records, str):
            raise ValueError(
                f"records must be a str for log record_format {self.record_format!r}"
            )
        if not self.record_format.is_log_format and not isinstance(self.records, dict):
            raise ValueError(
                f"records must be a dict for table record_format {self.record_format!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_end_time_after_start_time(self) -> ResultDict:
        if self.start_time is not None and self.end_time is not None:
            start = datetime.datetime.fromisoformat(self.start_time)
            end = datetime.datetime.fromisoformat(self.end_time)
            if end < start:
                raise ValueError(
                    f"end_time {self.end_time!r} must not be before"
                    f" start_time {self.start_time!r}"
                )
        return self

    executed_qcdl: Qcdl | None = Field(
        default=None,
        description=(
            "The :class:`~dwave.gate.qcdl.qcdl_models.Qcdl` payload"
            " representing the program that was executed."
        ),
    )
