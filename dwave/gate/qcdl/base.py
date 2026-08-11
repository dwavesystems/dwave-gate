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

import abc
import numbers
from collections import defaultdict
from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING, Any

from .exceptions import QCDLInternalError

if TYPE_CHECKING:
    from .components import Procedure, QCDLModule
    from .qcdl_circuit import QCDLCircuit


class QCDLArgument(abc.ABC):
    """This enables passing arbitrary objects to the compiler as args/kwargs to
    a procedure"""

    @abc.abstractmethod
    def serialize(self) -> Any:
        """Convert the object into something (jsonifiable) that can go to the
        compiler."""
        raise NotImplementedError


class QCDLModuleContainerBase(QCDLArgument):
    """This lets you pass an object containing QCDLModules to a procedure"""

    @property
    @abc.abstractmethod
    def qcdl_modules(self) -> Sequence[QCDLModule]:
        """The QCDLModule(s) this container holds

        Returns:
            sequence of QCDLModule
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        names = ", ".join(m.qcdl_module_name for m in self.qcdl_modules)
        return f"<QCDLModuleContainerBase {names}>"

    def serialize(self) -> list[str]:
        return [m.qcdl_module_name for m in self.qcdl_modules]

    @property
    @abc.abstractmethod
    def _is_qcdl_module(self) -> bool:
        # The purpose of this method is to provide a way to determine the class
        # w/o hasattr or isinstance
        raise NotImplementedError

    @property
    def procedure(self) -> Procedure:
        proc = self.qcdl_modules[0].procedure
        if not proc:
            raise QCDLInternalError("qubits must be in a procedure")
        return proc

    @property
    def state(self) -> QCDLCircuit:
        return self.procedure.state

    def __str__(self) -> str:
        return self.name

    @property
    def op_key(self) -> str | None:
        """Intended to represent the contents of a QCDLModuleContainerBase
        besides its qcdl_modules, used for procedure names when this is an
        argument.
        """
        return None

    def set_procedure(self, new_proc: Procedure) -> None:
        """Rewrap the QCDLModule objects with the new procedure

        The default implementation assumes that `qcdl_modules` is a list.

        Args:
            new_proc (Procedure): the new Procedure
        """

        for idx, m in enumerate(self.qcdl_modules):
            if isinstance(m, QCDLModuleContainerBase) and not m._is_qcdl_module:
                m.set_procedure(new_proc)
            else:
                # QCDLModule's qcdl_modules is not mutable
                if not isinstance(self.qcdl_modules, MutableSequence):
                    raise QCDLInternalError(
                        f"the default implementation only supports list,"
                        f" not {type(self.qcdl_modules)} on {self}"
                    )

                self.qcdl_modules[idx] = m.from_rewrapping(m, new_proc=new_proc)


class VariableExpression(QCDLArgument):
    """A QCDL variable expression

    This is a compile-time expression.

    This adds basic expression evaluation on top of variables. There's
    technically no reason to use a Variable instead of this since an
    expression can be just a single variable.

    This can support any pythonic expression, but will be safer to use than
    `eval`. You may use Variable objects (e.g., qcdl_args) in the expression.

    For examples, see: https://pypi.org/project/simpleeval/


    Args:
        variable_expression (Any): a python variable_expression (converted to
            str internally if not already a string)
    """

    TYPE = "variable_expression"

    def __init__(self, variable_expression: Any) -> None:
        if isinstance(variable_expression, VariableExpression):
            variable_expression = variable_expression._variable_expression
        if not isinstance(variable_expression, str):
            variable_expression = str(variable_expression)
        self._variable_expression: str = variable_expression

    @property
    def variable_expression(self) -> str:
        return self._variable_expression

    def serialize(self) -> dict:
        return {
            "type": VariableExpression.TYPE,
            VariableExpression.TYPE: self.variable_expression,
        }

    @classmethod
    def deserialize(cls, val: dict) -> VariableExpression:
        if val["type"] != VariableExpression.TYPE:
            raise TypeError(f"{val} is not a {VariableExpression.TYPE}")
        return VariableExpression(val[VariableExpression.TYPE])

    def __str__(self) -> str:
        """Backticks for legibility when printed in QCDL"""
        return "`{}`".format(self.variable_expression)

    def _grouped(self) -> str:
        """Parenthesize the expression, to prevent unforeseen order of
        operations changes"""
        return "(" + self._variable_expression + ")"

    def _apply_operator(
        self,
        operator: str,
        operand: numbers.Real | VariableExpression,
        right: bool = False,
    ) -> VariableExpression:
        """Apply a mathematical operator to the expression"""
        if isinstance(operand, (numbers.Real, VariableExpression)):
            if isinstance(operand, VariableExpression):
                operand = operand._grouped()  # type: ignore[assignment]  # grouped; no extra ``
            inputs = [self._grouped(), operator, str(operand)]
            if right:
                inputs.reverse()
            val = VariableExpression("{} {} {}".format(*inputs))
            return val
        else:
            raise TypeError(
                f"Cannot apply {operator} to operands that are not"
                f" Expressions or numbers.Real in this case, {type(operand)}"
            )

    def __add__(self, operand: numbers.Real | VariableExpression) -> VariableExpression:
        return self._apply_operator("+", operand)

    def __sub__(self, operand: numbers.Real | VariableExpression) -> VariableExpression:
        return self._apply_operator("-", operand)

    def __mul__(self, operand: numbers.Real | VariableExpression) -> VariableExpression:
        return self._apply_operator("*", operand)

    def __truediv__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("/", operand)

    def __floordiv__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("//", operand)

    def __mod__(self, operand: numbers.Real | VariableExpression) -> VariableExpression:
        return self._apply_operator("%", operand)

    def __radd__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("+", operand, right=True)

    def __rsub__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("-", operand, right=True)

    def __rmul__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("*", operand, right=True)

    def __rtruediv__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("/", operand, right=True)

    def __rfloordiv__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("//", operand, right=True)

    def __rmod__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("%", operand, right=True)

    def __iadd__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("+", operand)

    def __isub__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("-", operand)

    def __imul__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("*", operand)

    def __itruediv__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        # May be missing an edge case?
        return self._apply_operator("/", operand)

    def __ifloordiv__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("//", operand)

    def __imod__(
        self, operand: numbers.Real | VariableExpression
    ) -> VariableExpression:
        return self._apply_operator("%", operand)


class Variable(QCDLArgument):
    """This is the QCDL approach for creating a QCDL Variable

    Corresponds 1-to-1 with the qcdl_objects.py Variable. There will be
    no reason to use this instead of an RegisterExpression, but it's supported anyway
    due to its special handling in the compiler.

    Args:
        variable (str): Name of the variable
    """

    TYPE = "variable"

    def __init__(self, variable: str) -> None:
        self._variable = variable

    @property
    def variable(self) -> str:
        return self._variable

    def serialize(self) -> dict[str, Any]:
        return {"type": Variable.TYPE, Variable.TYPE: self.variable}


class IndexerMixin:
    """This is used as a mixin for Procedure and QCDLCircuit"""

    def __init__(self, next_indices: dict[str, int] | None = None) -> None:
        self._next_indices: defaultdict[str, int] = defaultdict(lambda: 0)
        if next_indices:
            self._next_indices.update(next_indices)

    def get_next_index(self, name: str) -> int:
        """Unique indices

        The uniqueness of the index depends on how this class is used.

        There are many contexts (labels, axes, memory address tags are examples)
        where a user needs a unique name, so a typical approach is to add a
        unique integer to a user provided name. The degree of uniqueness
        required depends on the context. For example, qcdl labels only need to
        be unique within a procedure, while axis ids need to be globally unique.

        This method will never return the same index twice for a given name.

        Args:
            name (str): namespace for the index

        Returns:
            int: index unique for this name
        """
        index = self._next_indices[name]
        self._next_indices[name] += 1
        return index
