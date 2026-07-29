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

import ast
import datetime
import logging
import token
import tokenize
from collections.abc import Generator
from enum import Enum
from functools import cached_property
from io import StringIO
from typing import Any, ClassVar

from .constants import INT_TO_FLOAT
from .exceptions import QCDLInternalError, QCDLRuntimeError

logger = logging.getLogger(__name__)


class RecordOutputToken(Enum):
    """These are the tokens that we allow to define a shape in RecordOutput"""

    ARRAY_START = "["
    ARRAY_END = "]"
    TUPLE_START = "("
    TUPLE_END = ")"

    RESULT = "Result"
    BOOLEAN = "Boolean"
    INTEGER = "Integer"
    DOUBLE = "Double"

    # Literal is a value that is known at compile time, e.g., a string description
    LITERAL = "Literal"

    # Error is a string that's handled as an exception message
    ERROR = "Error"
    # Exit includes Error
    EXIT = "Exit"

    @cached_property
    def is_primitive(self) -> bool:
        return self in [
            RecordOutputToken.RESULT,
            RecordOutputToken.BOOLEAN,
            RecordOutputToken.INTEGER,
            RecordOutputToken.DOUBLE,
        ]

    @cached_property
    def is_literal(self) -> bool:
        return self == RecordOutputToken.LITERAL

    @cached_property
    def is_error(self) -> bool:
        return self == RecordOutputToken.ERROR

    @cached_property
    def is_exit(self) -> bool:
        return self in [RecordOutputToken.ERROR, RecordOutputToken.EXIT]

    @cached_property
    def is_meta(self) -> bool:
        return self.is_literal or self.is_exit

    @cached_property
    def is_data(self) -> bool:
        return self.is_primitive or self.is_meta

    @cached_property
    def is_start(self) -> bool:
        return self in [RecordOutputToken.ARRAY_START, RecordOutputToken.TUPLE_START]

    @cached_property
    def is_end(self) -> bool:
        return self in [RecordOutputToken.ARRAY_END, RecordOutputToken.TUPLE_END]

    def convert_value(self, value: int) -> int | bool | float:
        if self in [RecordOutputToken.INTEGER, RecordOutputToken.RESULT]:
            return value
        elif self == RecordOutputToken.BOOLEAN:
            return bool(value)
        elif self == RecordOutputToken.DOUBLE:
            return value * INT_TO_FLOAT
        else:
            raise ValueError(f"{self} can not be used to convert {value}")


class RecordOutput:
    """Build a shape to store results into.

    One of these options could be built up like this:
        ro = RecordOutput()
        ro.array_start()
        ro.tuple_start()

        sval = "my literal"
        ro.literal(sval)
        ro.result("my_res")
        ro.boolean()

        ro.tuple_end()
        ro.array_end()

    This shape takes 2 values, a "result" and a boolean. In post processing, 2
    values from the data from the qubit (the result record) will be mapped into
    this shape, and the shape returned to the user.

    We support a couple ways to return the data to the user. The "log" format is
    matches what Microsoft currently supports. The other formats are speculative
    on my part for what I think would be useful when debugging.
    """

    def __init__(self) -> None:
        self._tokens: list[RecordOutputToken] = []
        self._names: list[str | None] = []
        self._metas: list[Any] = []

        self._num_primitives: int = 0
        self._is_balanced: bool = True
        self._error_message: str | None = None

    def _append_token(self, token: RecordOutputToken) -> None:
        self._tokens.append(token)
        # cache values so that we don't need to redetermine them every time we
        # load new values.
        if token.is_primitive:
            self._num_primitives += 1
        if token.is_error and self._error_message is None:
            self._error_message = self.description
        self._is_balanced = self._determine_if_balanced()

    def array_start(self, name: str | None = None) -> None:
        self._append_token(RecordOutputToken.ARRAY_START)
        self._names.append(name)

    def array_end(self) -> None:
        self._append_token(RecordOutputToken.ARRAY_END)
        self._names.append(None)

    def tuple_start(self, name: str | None = None) -> None:
        self._append_token(RecordOutputToken.TUPLE_START)
        self._names.append(name)

    def tuple_end(self) -> None:
        self._append_token(RecordOutputToken.TUPLE_END)
        self._names.append(None)

    def result(self, name: str | None = None) -> None:
        self._append_token(RecordOutputToken.RESULT)
        self._names.append(name)

    def boolean(self, name: str | None = None) -> None:
        self._append_token(RecordOutputToken.BOOLEAN)
        self._names.append(name)

    def double(self, name: str | None = None) -> None:
        self._append_token(RecordOutputToken.DOUBLE)
        self._names.append(name)

    def integer(self, name: str | None = None) -> None:
        self._append_token(RecordOutputToken.INTEGER)
        self._names.append(name)

    def literal(self, value: Any, name: str | None = None) -> None:
        """This lets you embed a literal into your data structure.

        The main motivation is to enable something like a print statement from
        qcdl where you can pass a string from the qcdl to the output results.

        Args:
            value (Any): The value of the literal
            name (str | None, optional): For formats which support it, this
            will be e.g., the column header. Defaults to None.
        """
        self._append_token(RecordOutputToken.LITERAL)
        self._names.append(name)
        self._metas.append(value)

    def error(self, value: Any, name: str | None = None) -> None:
        self._names.append(name)
        self._metas.append(value)
        self._append_token(RecordOutputToken.ERROR)

    def exit(self, value: Any, name: str | None = None) -> None:
        self._names.append(name)
        self._metas.append(value)
        self._append_token(RecordOutputToken.EXIT)

    @cached_property
    def exit_code(self) -> tuple[RecordOutputToken, Any] | None:
        meta_tokens = list(filter(lambda t: t.is_meta, self._tokens))
        if not meta_tokens:
            return None
        for token_type in [RecordOutputToken.ERROR, RecordOutputToken.EXIT]:
            try:
                idx = meta_tokens.index(token_type)
                return token_type, self._metas[idx]
            except ValueError:
                pass
        return None

    @property
    def error_message(self) -> str | None:
        return self._error_message

    def check_error(self, raise_if_error: bool = False) -> bool:
        error_message = self._error_message
        if error_message:
            if raise_if_error:
                raise QCDLRuntimeError(error_message)
            else:
                logger.critical(error_message)
                return True
        else:
            return False

    @property
    def balanced(self) -> bool:
        """Do array and tuple starts match their ends"""
        return self._is_balanced

    def _determine_if_balanced(self) -> bool:
        """The implementation of balanced"""
        array_depth = 0
        tuple_depth = 0
        for t in self._tokens:
            if t == RecordOutputToken.ARRAY_START:
                array_depth += 1
            elif t == RecordOutputToken.ARRAY_END:
                array_depth -= 1
            elif t == RecordOutputToken.TUPLE_START:
                tuple_depth += 1
            elif t == RecordOutputToken.TUPLE_END:
                tuple_depth -= 1

            if array_depth < 0:
                return False
            if tuple_depth < 0:
                return False

        if array_depth != 0:
            return False
        if tuple_depth != 0:
            return False

        return True

    @staticmethod
    def from_list(val: list[Any]) -> RecordOutput:
        """Read the serialized data structure."""
        ro = RecordOutput()
        for item in val:
            if isinstance(item, (list, tuple)):
                token_type, arg = item[0], item[1:]
                getattr(ro, token_type.lower())(*arg)
            else:
                getattr(ro, item.lower())()

        return ro

    @staticmethod
    def from_string(v: str) -> RecordOutput:
        """Construct a shape from a string.

        This is a convenience mechanism. The main reason I wrote it is to
        facilitate testing other code in this class, so this isn't really
        intended to be as flexible as building the shape programmatically.

        For example:
            "[(5,'Double', ('Boolean', 'my-literal'))]"

        To add a primitive, put in the data type, and for a literal, put in the
        value.
        """
        ro = RecordOutput()
        for toknum, tokval, _, _, _ in tokenize.generate_tokens(StringIO(v).readline):
            # print(toknum, tokval, token.tok_name[toknum])
            if toknum == token.OP:
                if tokval == "[":
                    ro.array_start()
                elif tokval == "]":
                    ro.array_end()
                elif tokval == ")":
                    ro.tuple_end()
                elif tokval == "(":
                    ro.tuple_start()
                elif tokval == ",":
                    continue
                else:
                    raise ValueError(f"unsupported OP {tokval}")
            elif toknum == token.NUMBER:
                ro.literal(ast.literal_eval(tokval))
            elif toknum == token.NAME:
                if tokval in ["True", "False"]:
                    ro.literal(ast.literal_eval(tokval))
                else:
                    getattr(ro, tokval.lower())()
            elif toknum == token.STRING:
                val = ast.literal_eval(tokval)
                if hasattr(ro, val.lower()):
                    getattr(ro, val.lower())()
                else:
                    ro.literal(val)
            elif toknum in [token.NEWLINE, token.ENDMARKER]:
                continue
            else:
                raise ValueError(
                    f"unsupported token type {token.tok_name[toknum]}"
                    f" with value {tokval}"
                )
        return ro

    def to_list(self) -> list[Any]:
        """This is the serialization that will go into the jmz."""
        data: list[Any] = []
        names = list(self._names)

        metas = list(self._metas)
        for t in self._tokens:
            name = names.pop(0)

            if t.is_meta:
                data.append((t.name, metas.pop(0), name))
            elif t.is_end:
                data.append(t.name)
            else:
                data.append((t.name, name))

        if names:
            raise QCDLInternalError(f"names not all used by to_list: {names}")
        if metas:
            raise QCDLInternalError(f"metas not all used by to_list: {metas}")

        return data

    @property
    def description(self) -> str:
        # use data types as values
        types_as_values = []
        for _, name, t in self.data():
            if t.is_primitive:
                # the type is the value
                types_as_values.append(t.value)

        key = self.make_key(types_as_values)
        if key is None:
            return "<empty shape>"
        return str(key)

    def __str__(self) -> str:
        return self.description

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, rhs: object) -> bool:
        if not isinstance(rhs, RecordOutput):
            return NotImplemented
        return self._tokens == rhs._tokens and self._names == rhs._names

    def __ne__(self, rhs: object) -> bool:
        return not (self == rhs)

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def data(
        self, values: Any = None, supply_missing_names: bool = False
    ) -> Generator[
        tuple[Any, str | tuple[Any, ...] | None, RecordOutputToken], None, None
    ]:
        """An iterator over all names, values, and metas

        The length of this iterator is the same as the number of tokens.

        Args:
            values (Any, optional): Sequence of primitive values to distribute
                across the tokens.  If ``None``, yielded values will be
                ``None``.  Defaults to None.
            supply_missing_names (bool, optional): When ``True``, tokens with
                no recorded name are assigned auto-generated names
                (``"value0"``, ``"meta0"``, etc.).  Defaults to False.

        Yields:
            tuple[Any, str | tuple[Any, ...] | None, RecordOutputToken]:
                ``(value_or_meta, name, token)`` — the resolved value (or
                ``None`` if *values* was not supplied), the name (a str, a
                tuple key, or ``None``), and the :class:`RecordOutputToken`.
        """
        if values is not None:
            self._check_values(values)

        val_idx = 0
        meta_idx = 0
        for raw_name, t in zip(self._names, self._tokens):
            if isinstance(raw_name, list):
                # json would convert tuples to lists, so convert to a tuple
                # so it can be a key
                name = tuple(raw_name)
            else:
                name = raw_name

            if t.is_primitive:
                # a primitive is something that comes from the qubit
                if name is None and supply_missing_names:
                    name = f"value{val_idx}"
                if values is not None:
                    value = values[val_idx]
                    if not isinstance(value, str):
                        value = t.convert_value(value)
                else:
                    value = None
                yield value, name, t
                val_idx += 1
            elif t.is_meta:
                # meta is like a value except that it's supplied at compile time
                meta = self._metas[meta_idx]
                if name is None and supply_missing_names:
                    name = f"meta{meta_idx}"
                yield meta, name, t
                meta_idx += 1
            else:
                yield t.value, name, t

        if meta_idx != len(self._metas):
            raise QCDLInternalError("metas not all used")
        if values is not None:
            if val_idx != len(values):
                raise QCDLInternalError("values not all used")

    @property
    def num_primitives(self) -> int:
        """This is how many values from the qubit are needed to fill in this
        shape."""
        # better performance to cache the value
        return self._num_primitives

    def _check_values(self, values: Any) -> None:
        if not self.balanced:
            raise ValueError("RecordOutput is not balanced " + str(self))

        if len(values) != self.num_primitives:
            raise ValueError(
                f"Require {self.num_primitives} values, received {len(values)}"
            )

    def make_key(self, values: Any) -> Any:
        """This is the final data format the user sees in their histogram.

        Values should be the length of primitives but may be of any data type.

        NOTE: I thought I'd be able to find something in ast or tokenize to do
        this work for me, but that's looking more complicated than just doing it
        manually like this.
        """
        self._check_values(values)
        key = ""
        for idx, (value, name, t) in enumerate(self.data(values=values)):
            if idx > 0:
                prev = self._tokens[idx - 1]
                need_comma = True

                if prev.is_start and t.is_start:
                    need_comma = False
                elif prev.is_end and t.is_end:
                    need_comma = False
                elif prev.is_start and t.is_data:
                    need_comma = False
                elif prev.is_data and t.is_end:
                    need_comma = False

                # this makes sure it stays a tuple and doesn't get converted to
                # a normal value
                if t == RecordOutputToken.TUPLE_END:
                    need_comma = True

                if need_comma:
                    key += ","

            if t.is_data:
                key += repr(value)
            else:
                key += t.value

        if not key:
            return None

        return ast.literal_eval(key)

    def values_as_log(
        self,
        values: Any,
        record_type: str | None = None,
        ensure_one: bool = True,
        record_format: RecordFormat | None = None,
        post_select: bool = True,
    ) -> list[dict[str, Any]] | None:
        """Format this shape into the data format requested by Microsoft

        There are numerous differences between vX.0 and vX.1. The only
        difference between v1.X and v2.X is whether names are supported.

        The current arrangement is that SenseData converts this list of dicts
        into a text file.

        Args:
            values (list(Any)): the values from the qubit
            format (RecordFormat): how to format the log
            record_type (str | None, optional): The record_type. Defaults to "RESULT".
            ensure_one (bool, optional): If True, then if this shape has no tokens
               it will at least create one entry. The use case is for START or END
               markers which may not have data. Defaults to False.
            post_select (bool): If True, data for shots that were post selected
               will be omitted from the returned data.

        Returns:
            list[dict]: the log entries
        """
        time_stamp = datetime.datetime.now().isoformat()

        if not self.balanced:
            raise ValueError("shape must be balanced for values_as_log")

        record_format = record_format or RecordFormat.QIR_V2_1
        if not record_format.is_log_format:
            raise ValueError(f"can not use {record_format} with values_as_log")

        is_point_1 = record_format == RecordFormat.QIR_V2_1
        if record_type is None:
            # the exact value here is mostly controlled by the compiler via the
            # name of the table
            if is_point_1:
                record_type = "OUTPUT"
            else:
                record_type = "RESULT"

        if not record_type:
            raise QCDLInternalError("record_type is required for the log format")

        def _rec_type_only() -> list[dict[str, Any]]:
            line = dict(record_type=record_type)
            if not is_point_1:
                # this is only used by v1
                line["time_stamp"] = time_stamp
            return [line]

        if ensure_one and not self._tokens:
            return _rec_type_only()

        lines = []

        # stack is used to sanity check that data structure starts match data
        # structure ends. it could be removed w/o affecting functionality.
        stack: list[Any] = []
        # iterate over all values, names, and tokens for the shape, filling in
        # the actual numerical data
        for value, name, t in self.data(values):
            if not t.is_literal and not t.is_end and stack and is_point_1:
                stack[-1]["value"] += 1

            if post_select and record_type == "END" and name == "post_select":
                if value:
                    # this shot is post selected!
                    return None
                else:
                    # just make sure we aren't dropping other data
                    if len(values) != 1:
                        raise QCDLInternalError(
                            f"expected exactly 1 value for post_select END,"
                            f" got {len(values)}"
                        )
                    # the post_select value shouldn't be returned to Azure
                    return _rec_type_only()

            if t.is_literal:
                # this used for the METADATA feature in vX.1
                lines.append(dict(record_type=name, value=value))
            elif t.is_data:
                lines.append(dict(record_type=record_type, value=value))
                if is_point_1:
                    type = t.name
                    if t == RecordOutputToken.BOOLEAN:
                        type = "BOOL"
                    elif t == RecordOutputToken.INTEGER:
                        type = "INT"

                    lines[-1]["type"] = type
                if name is not None:
                    # this is a v2.X feature
                    lines[-1]["name"] = name
            elif t.is_start:
                start_item: dict[str, Any] = dict(record_type=record_type)

                if is_point_1:
                    # just want ARRAY or TUPLE
                    start_item["type"] = t.name[:-6]
                    # this is a count of the number of items in this container
                    start_item["value"] = 0
                    start_item["name"] = name
                else:
                    start_item["value"] = t.name
                stack.append(start_item)
                lines.append(start_item)
            elif t.is_end:
                if not is_point_1:
                    lines.append(dict(record_type=record_type, value=t.name))
                stack.pop()

            if not is_point_1:
                # this is only used by v1
                lines[-1]["time_stamp"] = time_stamp

        if stack:
            raise QCDLInternalError(f"shape is not balanced, remaining stack: {stack}")

        return lines

    def values_as_key(self, values: Any) -> Any:
        return self.make_key(values)

    def values_as_dict(self, values: Any) -> dict[Any, Any]:
        """This is a useful format if you want to treat these as rows in a
        table. It ignores the tuples/lists.

        If the names weren't unique, the user will not receive all the data.
        """
        return {
            name: value
            for value, name, token in self.data(values, supply_missing_names=True)
            if token.is_data
        }

    def values_as_list(self, values: Any) -> list[Any]:
        return [value for value, _, token in self.data(values) if token.is_data]


class RecordFormat(Enum):
    """These are all the ways we support formatting the records field"""

    TABLE = "table"
    ARRAYS = "arrays"
    LIST = "list"
    KEY = "key"
    RAW = "raw"
    POLARS = "polars"
    QIR_V2_1 = "qir.v2.1"

    @property
    def is_log_format(self) -> bool:
        return self == RecordFormat.QIR_V2_1

    @property
    def is_table_format(self) -> bool:
        return self in [RecordFormat.TABLE, RecordFormat.POLARS]

    @staticmethod
    def from_record_format(val: str | RecordFormat | None) -> RecordFormat:
        if val is None:
            return RecordFormat.RAW
        elif val == "log":
            return RecordFormat.QIR_V2_1
        elif isinstance(val, RecordFormat):
            return val

        try:
            return RecordFormat[val.upper()]
        except KeyError:
            raise ValueError(f"record format type {val} is not supported")

    from_record_formatter = from_record_format
