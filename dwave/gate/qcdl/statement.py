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
from typing import Any, cast

from addict import Addict

from .exceptions import QCDLInternalError
from .qcdl_models import QCDLStatement
from .qcdl_objects import format_signature
from .utils import is_qubit_name, is_qubit_or_coupler_name

logger = logging.getLogger(__name__)


class Statement:
    def __init__(self, statement: dict | QCDLStatement):
        stmt: dict = (
            statement.model_dump(exclude_unset=True)
            if isinstance(statement, QCDLStatement)
            else statement
        )
        if not isinstance(stmt, dict):
            raise QCDLInternalError(
                f"unexpected statement type {type(stmt)} for {stmt}"
            )
        if not isinstance(stmt, Addict):
            stmt = Addict(stmt)
        addict_stmt: Any = cast(Any, stmt)
        self._orig = addict_stmt.to_dict()

        self.qubit_name = stmt.get("qubit")
        self.op = stmt["op"]
        self._kwargs = dict(stmt.get("kwargs", {}))
        self.if_name = addict_stmt.if_name

        self._modules: list[str] = []

        def _add_module(q: str) -> None:
            if q not in self._modules:
                self._modules.append(q)

        if self.qubit_name:
            # procedure_library don't use qubit_name
            self._modules.append(self.qubit_name)

        for q in stmt.get("qubits", []):
            _add_module(q)
        for q in self._kwargs.pop("qubits", []):
            _add_module(q)

        self._args = []
        for arg in stmt.get("args", []):
            if self.op not in ["comment", "If", "c_if", "c_if_else"] and (
                module_name := _get_module_from_arg(arg)
            ):
                if module_name not in self._modules:
                    # FIXME: procedures put qubits in as args
                    self._modules.append(module_name)
                # else:
                #     # it depends on the op whether this is a problem or not, so
                #     # rely on downstream code to determine that
                #     logger.debug(f"duplicate qubit {module_name} in {self._orig}")
            else:
                self._args.append(arg)

        if self.op in ["If", "Else", "Endif"]:
            # FIXME: in practice, we only use values to indicate qubits, but the
            # compiler does a fully recursive search
            for name in list(self._kwargs):
                value = self._kwargs[name]
                # condition is the one case where the value could be a qubit,
                # but where it represents a prior measurement of that qubit
                if name not in ["condition", "source_qubit"] and is_qubit_name(value):
                    self._modules.append(self._kwargs.pop(name))

        if len(self._modules) != len(set(self._modules)):
            raise QCDLInternalError(
                f"duplicate qubits found {self._modules} from {self._orig} as {self}"
            )

        if not self._modules:
            raise QCDLInternalError(
                f"statement {self} created with no qubits, {self._orig}"
            )

        if not self.unresolved_qubits:
            qargs = self.qargs
            if len(qargs) != len(set(qargs)):
                raise QCDLInternalError(
                    f"duplicate qargs {qargs} in {self} from {self._orig}"
                )

    @property
    def is_procedure_call(self) -> bool:
        return self.qubit_name is None

    @property
    def unresolved_qubits(self) -> list[str]:
        unresolved = []
        for module in self._modules:
            if not is_qubit_or_coupler_name(module):
                # could be a variable
                unresolved.append(module)
        return unresolved

    @property
    def caller_qubits(self) -> list[Any]:
        if self.is_procedure_call:
            return self._orig["qubits"]
        else:
            return self._modules

    @property
    def args(self) -> list[Any]:
        """The list of arguments used by this instruction, excluding qubit arguments"""
        return list(self._args)

    def reassign_arg(self, i: int, new_value: Any) -> None:
        self._args[i] = new_value

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self._kwargs)

    def reassign_kwarg(self, name: str, new_value: Any) -> None:
        self._kwargs[name] = new_value

    @property
    def modules(self) -> list[str]:
        """The list of qubits and couplers used by this instruction. This
        list does not include conditional statements where a qubit was used as
        the condition.
        """
        return self._modules

    @property
    def qubits(self) -> list[str]:
        return [q for q in self._modules if is_qubit_name(q)]

    @property
    def qargs(self) -> list[int]:
        return [int(q[1:]) for q in self.qubits]

    @property
    def cargs(self) -> None:
        return None

    @property
    def condition(self) -> Any:
        if self.op not in ["If", "c_if", "c_if_else"]:
            raise QCDLInternalError(f"op {self.op} doesn't have a condition")

        if self.op in ["c_if", "c_if_else"]:
            condition = self.args[-1]
        elif self.args:
            condition = self.args[0]
        elif self.kwargs and "condition" in self.kwargs:
            condition = self.kwargs["condition"]
        else:
            # don't rely on previously set signal
            raise QCDLInternalError(f"statement {self} doesn't have a condition")

        if isinstance(condition, dict):
            _cond: Any = condition
            if _cond.get("type") == "variable":
                return _cond.variable
            return _cond.value
        else:
            return condition

    def simple_desc(self) -> str:
        if self.op == "cpu":
            expr = self.args[0].strip().splitlines()
            return "; ".join(expr)
        elif self.op == "If":
            return f"If({self.condition})"
        elif self.op in ["Else", "Endif"]:
            goto = self.kwargs.get("goto")
            if goto:
                return f"{self.op}(goto={goto})"
            else:
                return self.op
        elif self.op in ["label", "goto"]:
            label = self.kwargs["label"]
            return f"{self.op}({label})"
        else:
            return str(self)

    def __str__(self) -> str:
        if self.is_procedure_call:
            return format_signature(
                self.op,
                qubits=[str(q) for q in self._modules],
                args=self.args,
                nargs=self.kwargs,
            ).strip()
        else:
            return format_signature(
                self.op,
                qubit=self.qubit_name,
                args=self._orig.get("args", []),
                nargs=self.kwargs,
            ).strip()


def _get_module_from_arg(arg: Any) -> str | None:
    if isinstance(arg, dict) and arg.get("type") == "variable":
        arg = arg.get("variable")

    if is_qubit_or_coupler_name(arg):
        return arg

    return None
