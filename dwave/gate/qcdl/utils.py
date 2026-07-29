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

import logging
import math
import re
from collections import abc
from collections.abc import Generator, Mapping, Sequence, Set
from fractions import Fraction
from typing import Any, Callable

import numpy as np

from .exceptions import QCDLUserError

logger = logging.getLogger(__name__)


def is_qubit_name(name: object) -> re.Match[str] | None:
    if not isinstance(name, str):
        return None
    return re.match(r"^q(\d+)$", name)


def is_qubit_or_coupler_name(name: object) -> re.Match[str] | None:
    if not isinstance(name, str):
        return None
    return re.match(r"^[cq](\d+)$", name)


def iteritems(mapping: Any) -> Any:
    return getattr(mapping, "iteritems", mapping.items)()


def objwalk(
    obj: Any,
    path: tuple[Any, ...] = (),
    memo: set[int] | None = None,
    containers: bool = False,
    sort_items: bool = True,
) -> Generator[tuple[Any, Any], None, None]:
    """Walk python objects recursively

    Based on this code:
    http://code.activestate.com/recipes/577982-recursively-walk-python-objects/

    NOTE: Set types are yielded in the order determined by the enumerate method

    Args:
        obj (Any): container object
        path (tuple[Any, ...], optional): path root. Defaults to ().
        memo (set[int] | None, optional): seen objects. Defaults to None.
        containers (bool, optional): yield the containers too.
            Defaults to False.
        sort_items (bool, optional): yield dict items in sorted order.
            Defaults to True.

    Yields:
        Any: object in the container
    """
    if memo is None:
        memo = set()

    iterator: Any = None
    if isinstance(obj, Mapping):
        if sort_items:

            def iterator(m: Any) -> Any:  # type: ignore[assignment]
                return sorted(iteritems(m))
        else:
            iterator = iteritems
    elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, str):
        iterator = enumerate

    if iterator is not None:
        if containers:
            # yield the containers, too
            yield path, obj

        if id(obj) not in memo:
            memo.add(id(obj))
            for path_component, value in iterator(obj):
                for result in objwalk(
                    value, path + (path_component,), memo, containers=containers
                ):
                    yield result
            memo.remove(id(obj))
    else:
        yield path, obj


def map_container(
    obj: Any,
    map_value: Callable[[Any], Any],
    map_key: Callable[[str], str] | None = None,
    map_object: Callable[[Any], None] | None = None,
    map_value_instance_types: type | tuple[type, ...] = object,
    map_value_in_dict: Callable[[Any], Any] | None = None,
) -> None:
    map_value_in_dict = map_value_in_dict or map_value
    memo = set()
    for path, value in objwalk(obj, containers=True):
        # for sets and dicts, we have to apply the map to all
        # elements at the same time
        if isinstance(value, abc.MutableSet):
            # Order doesn't matter, so no effective change if the mapping is
            # within the original set of qubits
            new_set = {map_value(v) for v in value}
            value.clear()
            value |= new_set
            memo.add(id(value))
            memo.update([id(v) for v in new_set])
        elif isinstance(value, abc.MutableMapping) and map_key:
            # change the names of any keys. only known case
            # are qubit names in arbitrary_functions state_data
            # the values will get mapped elsewhere in the recursion
            new_dict = {map_key(k): map_value_in_dict(v) for k, v in value.items()}
            value.clear()
            value.update(new_dict)
            memo.add(id(value))
        elif id(value) in memo:
            continue
        elif isinstance(value, map_value_instance_types):
            try:
                new_value = map_value(value)
            except QCDLUserError as e:
                raise QCDLUserError(f"Failed to map {value} from path {path}") from e

            if value is new_value:
                continue

            parent = obj
            for elem in path:
                if isinstance(parent, abc.MutableSet):
                    if id(parent) not in memo:
                        logger.warning(f"possibly not mapped {parent}")
                    break

                child = parent[elem]
                if child is value:
                    parent[elem] = new_value
                else:
                    parent = parent[elem]
        elif isinstance(value, abc.MutableSequence):
            continue
        elif map_object:
            map_object(value)


def simplify_float(f: float, threshold: float = 1e-10) -> str:
    """Convert a float to a string representation

    Mostly the intent of this method is to find a value as a fraction of pi, if
    it exists.

    Args:
        f (float): value
        threshold (float, optional): If not a fraction of pi, return value
            with this level of accuracy. Defaults to 1e-10.

    Returns:
        str: A simplified representation
    """
    if math.isnan(f):
        return str(f)
    if abs(round(f) - f) <= threshold:
        return "%i" % f

    ratio = Fraction(f / np.pi).limit_denominator()
    if abs(ratio.numerator) < 100:
        rep = "pi"
        if ratio.numerator == -1:
            rep = "-" + rep
        elif ratio.numerator != 1:
            rep = str(ratio.numerator) + "*" + rep
        if ratio.denominator != 1:
            rep = "%s/%i" % (rep, ratio.denominator)
        return rep

    if threshold:
        return str(round(f, int(-math.log10(threshold))))
    else:
        return str(f)
