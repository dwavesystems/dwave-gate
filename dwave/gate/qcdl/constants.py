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

from __future__ import annotations

import enum

# conversion factors
FLOAT_TO_INT = 2**16
INT_TO_FLOAT = 1.0 / FLOAT_TO_INT

# these are the highest and lowest values allowed on the qubit. exceeding these
# would overflow on the qubit.
MIN_INT_REGISTER_VALUE = -(2**17)
MAX_INT_REGISTER_VALUE = 2**17 - 1
UNSIGNED_MAX_INT_REGISTER_VALUE = 2**18 - 1

MIN_FLOAT_REGISTER_VALUE = -2
MAX_FLOAT_REGISTER_VALUE = MAX_INT_REGISTER_VALUE * INT_TO_FLOAT

NUM_RECS = 4


class LogicalOutcomeToInteger(enum.IntEnum):
    """Maps the numerical encoding to an erasure bitstring representation."""

    ZERO = 0
    ONE = 1
    SPLAT = -1

    @staticmethod
    def from_outcome(val: str | int | float) -> LogicalOutcomeToInteger:
        """Instantiate a :class:`.LogicalOutcomeToInteger` from various inputs.

        Args:
            val (str | int | float): Input value.

        Returns:
            :class:`.LogicalOutcomeToInteger`: Erasure bitstring representation.
        """
        if val in ["0", 0]:
            return LogicalOutcomeToInteger.ZERO
        elif val in ["1", 1]:
            return LogicalOutcomeToInteger.ONE
        elif val in ["-1", -1, "2", 2, "*", "p"]:
            # backwards compatibility to allow the previous value of 2
            return LogicalOutcomeToInteger.SPLAT
        else:
            raise ValueError(f"unknown Logical Measurement type {val=}")

    @property
    def as_bit(self) -> str:
        """Return the erasure bitstring representation for the numerical encoding.

        Returns:
            str: A "0", "1", or "*".
        """
        if self == LogicalOutcomeToInteger.SPLAT:
            return "*"
        else:
            return str(self.value)
