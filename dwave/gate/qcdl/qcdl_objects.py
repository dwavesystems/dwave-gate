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

"""
QCDL elements, originally in the compiler, used by both QCDL and lark transformer
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

import numpy as np

from .base import VariableExpression

# -----------------------------------------------------------------------------
# custom QCDL parse tree elements
from .utils import simplify_float


def format_signature(
    op: str,
    qubit: str | None = None,
    card: str | None = None,
    qubits: Any = None,
    args: Any = None,
    nargs: Any = None,
    simplify: bool = True,
    indent: str | int = "",
) -> str:
    if qubit:
        qubit = "{qubit}.".format(qubit=qubit)
        if card:
            qubit = "{qubit}{card}.".format(qubit=qubit, card=card)
    else:
        qubit = ""

    if isinstance(indent, int):
        indent = " " * indent

    def handle_variable(v: Any, prevent_quotes: bool = False) -> str:
        if isinstance(v, dict) and v.get("type") == "variable_expression":
            v = VariableExpression.deserialize(v)

        if isinstance(v, Variable):
            return v.name
        elif isinstance(v, VariableExpression):
            return str(v)
        elif prevent_quotes or (isinstance(v, str) and re.match(r"q\d+$", v)):
            return v
        elif isinstance(v, dict) and v.get("type") == "cpu":
            return '"{v}"'.format(v=v["value"])
        elif isinstance(v, str):
            if "\n" in v:
                return '"""{v}"""'.format(v=textwrap.indent(v, indent + "   "))
            else:
                return '"{v}"'.format(v=v)
        elif isinstance(v, np.ndarray):
            # e.g., error "gates" include numpy matrices
            v = repr(v).replace(" ", "")
            return v
        elif v and isinstance(v, set):
            # for reproducibility when comparing qcdls
            return "{%s}" % (", ".join(sorted([str(x) for x in v])))
        elif hasattr(v, "name"):
            # e.g., an element
            return '"{v}"'.format(v=v.name)
        elif simplify and isinstance(v, float):
            return simplify_float(v)
        else:
            return str(v)

    sig = []

    if qubits:
        qubits = ", ".join([handle_variable(q, prevent_quotes=True) for q in qubits])
        qubits = "[{qubits}]".format(qubits=qubits)
        sig.append(qubits)

    if args:
        sig.extend([handle_variable(a) for a in args])
    if nargs:
        # preserve order of phi and lam!
        # sorted_nargs = sorted(nargs.items(), key=lambda x: x[0])
        sig.extend(
            ["{k}={v}".format(k=k, v=handle_variable(v)) for k, v in nargs.items()]
        )
    txt = "{qubit}{op}({sig})\n".format(qubit=qubit, op=op, sig=", ".join(sig))
    return indent + txt


def get_indent_change(stmt: Any) -> int | None:
    if isinstance(stmt, ParserAsmInstruction) and stmt.opdat["op"] == "qcdl_indent":
        return stmt.opdat["args"][0]
    else:
        return None


def compute_indentation(indent_change: int, indent: str) -> str:
    if not isinstance(indent_change, int):
        raise ValueError(f"indentation change must be an integer, not {indent_change}")
    if indent_change > 0:
        return indent + " " * indent_change
    else:
        return indent[:indent_change]


class Variable:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return "<variable: %s>" % self.name

    def _json(self) -> dict[str, Any]:
        return dict(type="variable", value=self.name)

    __repr__ = __str__


class BeginStatement:
    def __init__(self, items: Any) -> None:
        self.stype = items[0].value
        # self.stype = items[1].value
        if len(items) > 1:
            self.options = items[1]
        else:
            self.options = None

    def __str__(self) -> str:
        return "<Begin: %s (%s)>" % (self.stype, self.options)

    def _json(self) -> str:
        return str(self)

    __repr__ = __str__


class EndStatement:
    def __init__(self, items: Any) -> None:
        self.stype = items[0].value
        # self.stype = items[1].value

    def __str__(self) -> str:
        return "<End: %s>" % (self.stype)

    def _json(self) -> str:
        return str(self)

    __repr__ = __str__


class ParserAsmProgram:
    def __init__(
        self, items: Any = None, options: Any = None, prog: Any = None
    ) -> None:
        try:
            self.options = options or {}
            self.prog = prog or items[:]
        except Exception as err:
            raise ValueError(
                f"Failed in ParserAsmProgram, items={items}, err={err}"
            ) from err

    def __str__(self) -> str:
        return "<ParserAsmProgram: options=%s, prog=%s lines>" % (
            self.options,
            len(self.prog),
        )

    def _json(self) -> dict[str, Any]:
        return {"description": str(self), "prog": self.prog}

    def _qcdl(self, indent: str = "") -> str:
        txt = ""
        for stmt in self.prog:
            indent_change = get_indent_change(stmt)
            if indent_change is not None:
                indent = compute_indentation(indent_change, indent)
                continue

            txt += stmt._qcdl(indent=indent)
        return txt

    __repr__ = __str__


class AsmSync:
    def __init__(self, qubits: Any, nargs: dict[str, Any] | None = None) -> None:
        nargs = nargs or {}
        self.qubits = qubits
        self.nargs = nargs

    def __str__(self) -> str:
        return "<AsmSync: qubits=%s nargs=%s>" % (self.qubits, self.nargs)

    def _json(self) -> dict[str, Any]:
        return {"description": str(self)}

    def _qcdl(self, indent: str = "") -> str:
        if self.nargs:
            qubits = sorted(self.qubits)

            return format_signature(
                op="sync",
                qubit=qubits[0],
                args=qubits[1:],
                nargs=self.nargs,
                indent=indent,
            )
        else:
            return indent + "- %s -\n" % (" ".join(sorted(self.qubits)))

    __repr__ = __str__


class ProcedureCallStatement:
    def __init__(self, items: Any) -> None:
        self.procedure_name = items[0].value

        if type(items[1]).__name__ == "Tree" and hasattr(items[1], "children"):
            # this code is only used by the compiler when parsing QCDLv2
            # items[1] is a lark Tree object
            self.qubits = items[1].children
        elif isinstance(items[1], list):
            self.qubits = items[1]
        else:
            self.qubits = [items[1]]

        self.args = items[2]
        self.nargs = items[3]

    def __str__(self) -> str:
        return "<ProcedureCallStatement: %s.%s %s %s>" % (
            self.procedure_name,
            self.args,
            self.nargs,
            self.qubits,
        )

    def _json(self) -> dict[str, Any]:
        return {
            "procedure_name": str(self.procedure_name),
            "qubits": self.qubits,
            "args": self.args,
            "named_args": self.nargs,
        }

    def _qcdl(self, indent: str = "") -> str:
        return format_signature(
            op=self.procedure_name,
            qubits=self.qubits,
            args=self.args,
            nargs=self.nargs,
            indent=indent,
        )

    __repr__ = __str__


class ParserAsmInstruction:
    def __init__(self, items: Any, card_name: str | None = None) -> None:
        try:
            self.qubit_name = items[0].value
            self.opdat = items[1]
            self.card_name = card_name
        except Exception as err:
            raise ValueError(
                f"Failed in ParserAsmInstruction, items={items}, err={err}"
            ) from err

    def __str__(self) -> str:
        args = ", ".join(["%s=%s" % kv for kv in self.opdat.items()])
        return "<ParserAsmInstruction: %s.%s %s>" % (
            self.qubit_name,
            self.card_name,
            args,
        )

    def _json(self) -> str:
        return str(self)

    def _qcdl(self, indent: str = "") -> str:
        op = self.opdat["op"]
        args = self.opdat.get("args")
        nargs = self.opdat.get("named_args")
        if op == "comment" and not nargs.get("data"):
            txt = args[0]
            if not txt:
                return "\n"
            else:
                return indent + "# " + txt + "\n"

        return format_signature(
            op=op,
            qubit=self.qubit_name,
            card=self.card_name,
            args=args,
            nargs=nargs,
            indent=indent,
        )

    __repr__ = __str__
