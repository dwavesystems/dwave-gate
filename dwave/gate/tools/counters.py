# Copyright 2022 D-Wave Systems Inc.
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

__all__ = [
    "IDCounter",
]

import itertools
import warnings
from typing import Optional


class IDCounter:
    """ID number counter.

    Generates pseudo-random alphanumeric ID numbers with a certain length in batches. If all ID
    numbers have been used in a batch, a new batch is generated. If all ID numbers of the declared
    length has been used, the length is incremented by one and ID numbers of the new length is
    generated instead.
    """

    length = _default_length = 4
    batch = _default_batch = 1000

    # alphanumeric string containing digits 0-9 and letters a-z to generate IDs;
    # starts with 'q' for quantum, mixes digits and letters to look more random
    _alphanum = "q0a1b2c3d4e5f6g7h8i9jklmnoprstuvwxyz"
    _id_gen = itertools.combinations(_alphanum, r=length)

    id_set = set()

    @classmethod
    def next(cls) -> str:
        """Returns a semi-random (unique) alphanumeric ID."""
        if not cls.id_set:
            cls.refresh()
        return cls.id_set.pop()

    @classmethod
    def refresh(cls) -> None:
        """Refreshes the set of available ID numbers; automatically done when necessary."""
        for _ in range(cls.batch):
            try:
                cls.id_set.add("".join(next(cls._id_gen)))

            except StopIteration:
                cls.length += 1
                if cls.length > len(cls._alphanum):
                    if not cls.id_set:
                        raise ValueError(
                            "ID length cannot be longer than number of unique characters available."
                        )
                    warnings.warn(
                        f"Insufficient characters to generate unique ID of length {cls.length} or "
                        f"longer. Generated batch of {len(cls.id_set)} instead of requested "
                        f"batch size of {cls.batch} IDs"
                    )
                    break

                cls._id_gen = itertools.combinations(cls._alphanum, r=cls.length)
                cls.id_set.add("".join(next(cls._id_gen)))

                continue

    @classmethod
    def reset(cls, length: Optional[int] = None, batch: Optional[int] = None) -> None:
        """Resets the ID counter to use a certain length and/or batch size.

        Args:
            length: The (initial) length of unique ID numbers.
            batch: The size of each generated batch of ID numbers. The lower the number, the less
                variation there will be between IDs; the higher the number, the more variation
                there will be, but with a higher memory and performance impact at refresh time.
        """
        cls.length = length or cls._default_length
        cls.batch = batch or cls._default_batch

        cls._id_gen = itertools.combinations(cls._alphanum, r=cls.length)

        cls.id_set = set()
