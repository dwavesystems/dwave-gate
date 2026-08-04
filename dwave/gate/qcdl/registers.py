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

"""QCDL supports integer and fixed-point registers you can use for simple
classical expressions (e.g., multiplication and XOR), and comparisons. The
:ref:`qcdl_basic_registers_arithmetic` section introduces registers.
"""

from __future__ import annotations

import abc
import inspect
import keyword
import re
from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Type

import numpy as np

from .base import QcdlArgument, QcdlModuleContainerBase
from .constants import (
    FLOAT_TO_INT,
    INT_TO_FLOAT,
    MAX_FLOAT_REGISTER_VALUE,
    MAX_INT_REGISTER_VALUE,
    MIN_FLOAT_REGISTER_VALUE,
    MIN_INT_REGISTER_VALUE,
    UNSIGNED_MAX_INT_REGISTER_VALUE,
)
from .exceptions import QCDLInternalError, QCDLUserError
from .utils import is_qubit_or_coupler_name

if TYPE_CHECKING:
    from .components import Procedure, QcdlModule

operators_to_math = {
    # special
    "__ilshift__": "<<=",
    "__neg__": "-",
    # math
    "__add__": "+",
    "__sub__": "-",
    "__rsub__": "-",
    "__mul__": "*",
    "__iadd__": "+=",
    "__isub__": "-=",
    "__imul__": "*=",
    # bitwise operators
    "__and__": "&",
    "__xor__": "^",
    "__or__": "|",
    "__rshift__": ">>",
    # "__lshift__": "<<",
    # comparisons
    "__eq__": "==",
    "__ne__": "!=",
    "__lt__": "<",
    "__gt__": ">",
    "__le__": "<=",
    "__ge__": ">=",
}


def validate_value(value: Any, dtype: str | type, signed: bool = True) -> None:
    """Check if the specified value can be assigned to a register.

    .. note:: This is a convenience function you can use from Python to check
        your QCDL program.

    Raises:
        QCDLUserError: If the assignment is not supported.

    Args:
        value: Value or array of values to validate.
        dtype: int or float.
        signed: Only integers may be unsigned.

    Examples:

        .. testcode::

            from dwave.gate.qcdl.registers import validate_value

            validate_value(1.5, dtype=float)

    """
    if dtype is int:
        dtype = "int"
    elif dtype is float:
        dtype = "float"

    if dtype not in ["float", "int"]:
        raise QCDLUserError(f"{dtype} is not a valid dtype")

    seq = to_sequence(value)
    if any([isinstance(v, bool) for v in seq]):
        # bools are instances of ints, so don't allow coercing them
        raise QCDLUserError("may not assign bool to register")

    # convert value to an array if it isn't one already
    value = np.array(seq)
    if not (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ):
        ok_coerce = False
    elif dtype == "int":
        # don't allow a non-whole-number to be coerced to an int
        ok_coerce = np.issubdtype(value.dtype, np.integer) or all(
            [v.is_integer() for v in value]
        )
    else:
        ok_coerce = True

    if not ok_coerce:
        raise QCDLUserError(f"may not assign value {value} to a {dtype} register")

    # NOTE: this only checks the initial value, it can't check overflows
    # that could happen on the qubit
    if dtype == "float":
        low = MIN_FLOAT_REGISTER_VALUE
        high = MAX_FLOAT_REGISTER_VALUE
    elif signed:
        low = MIN_INT_REGISTER_VALUE
        high = MAX_INT_REGISTER_VALUE
    else:
        low = 0
        high = UNSIGNED_MAX_INT_REGISTER_VALUE

    bad_values = value[(value < low) | (value > high)]
    if len(bad_values) > 0:
        bad_values = ", ".join(map(str, bad_values))
        raise QCDLUserError(
            f"{dtype} register value {bad_values} is not in range [{low}, {high})"
        )


def sanitize_name(name: str) -> str:
    # make it look like a variable name
    return re.sub(r"[^a-zA-Z\d_]+", "_", str(name))


def validate_name(name: str) -> None:
    """Check if the specified string is valid as a name for a register.

    Args:
        name: Proposed register name.

    Raises:
        QCDLUserError: If the string is not supported for a register name.

    Examples:

        .. testcode::

            from dwave.gate.qcdl.registers import validate_name

            validate_name("r0")
    """
    if (
        not isinstance(name, str)
        or name in Output._outputs
        or keyword.iskeyword(name)
        or not name.isidentifier()
        or is_qubit_or_coupler_name(name)
    ):
        raise QCDLUserError(f"{name} is not a valid name for a register")


def to_sequence(obj: Any) -> np.ndarray | Sequence[Any]:
    """Convert to a list if not already a list

    If obj was not a Sequence or ndarray, then this will convert it to a list.
    Otherwise it will return back the same object.
    """
    return (
        obj
        if (isinstance(obj, (np.ndarray, Sequence)) and not isinstance(obj, str))
        else [obj]
    )


def validate_memory_allocation(
    name: str, initial_value: Any, length: int, dtype: str | type, signed: bool = True
) -> None:
    validate_name(name)

    values = to_sequence(initial_value)

    if len(values) != length:
        raise QCDLUserError(f"Initial value {values} has wrong length {length}")

    # MemoryContents can only hold 1024 per block, but that resource is shared
    # across all registers (and some are automatically claimed by the compiler).
    # The number of unallocated registers can only be determined by the compiler,
    # so 1000 is an upper bound on what might be available.
    if length > 1000:
        raise QCDLUserError(f"Array {name} length {length} is too long")

    for value in values:
        validate_value(value, dtype, signed=signed)


class RegisterInitializerMixin:
    # Type declarations for attributes set in _register_initialization.
    _modules: Any
    _value: str
    can_read: bool
    length: int
    dtype: str
    scope_id: int | None

    def _register_initialization(
        self,
        modules: Sequence[QcdlModule],
        initial_value: Any,
        dtype: str | type,
        name: str | None = None,
        length: int | None = None,
        alias: bool | str = False,
        ignore_reallocation: bool = False,
        signed: bool = True,
        scope_id: int | None = None,
    ) -> None:
        """Make sure all registers are initialized the same way. The docstrings
        for the arguments here are found in the Register and FixedPointRegister
        classes.
        """
        modules = to_sequence(modules)  # type: ignore[assignment]
        state = modules[0].procedure.state

        if isinstance(initial_value, np.ndarray):
            initial_value = initial_value.tolist()

        if length is None:
            length = len(initial_value) if isinstance(initial_value, Sequence) else 1

        if name is None:
            name = "{dtype}{idx}".format(
                dtype=dtype, idx=state.get_next_index("register")
            )

        dtype = str(dtype)
        validate_memory_allocation(
            name, initial_value=initial_value, length=length, dtype=dtype, signed=signed
        )

        if alias is not True:
            # use alias=True if some other code called allocate_memory for this
            # register
            q, kwargs = prepare_multiqubit_stmt(modules)
            if not isinstance(alias, (bool, str)):
                raise QCDLUserError(
                    f"an alias must be a bool or the name of another"
                    f" register, not {alias} which has type {type(alias)}"
                )
            if not isinstance(alias, bool):
                # A string value for alias allows us to create a new register
                # which points to a pre-existing other register. This supports
                # type punning.
                kwargs["alias_of"] = alias
            q.allocate_memory(
                name,
                initial_value=initial_value,
                block=0,
                length=length,
                dtype=dtype,
                ignore_reallocation=ignore_reallocation,
                scope_id=scope_id,
                **kwargs,
            )

        self._modules = modules
        self._value = name
        self.can_read = True
        self.length = length
        self.dtype = dtype
        self.scope_id = scope_id

    @property
    def value(self) -> str:
        return self._value

    @property
    def modules(self) -> Sequence[QcdlModule]:
        return self._modules


def prepare_multiqubit_stmt(
    modules: Sequence[QcdlModule],
) -> tuple[QcdlModule, dict[str, Any]]:
    """Support for conversion of a set of identical statements on multiple
    qubits into one statement on a single qubit using the qubits kwarg"""
    other_qubits = [m for m in modules[1:]]
    kwargs = {}
    if other_qubits:
        kwargs["qubits"] = other_qubits
    return modules[0], kwargs


class OpsMixin:
    # These attributes are declared here for mypy's benefit; concrete subclasses
    # (e.g., Register, FixedPointRegister) provide them via mixins.
    value: Any
    can_read: bool
    modules: Any
    master_kwargs: Any
    scope_id: Any

    def __str__(self) -> str:
        return "<{cat} {val}>".format(cat=self.__class__.__name__, val=self.value)

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def _get_value(val: Any, for_read: bool = True, add_parens: bool = True) -> str:
        """Prepare the object to be included in an expression

        Args:
            val ([OpsMixin, int, float]): The value
            for_read (bool, optional): If True, will verify
                it's not an Output. Defaults to True.
            add_parens (bool, optional): If True, will add
                parenthesis to enforce order of operations. Defaults to True.

        Raises:
            QCDLUserError: the value may not be read

        Returns:
            str: A string object
        """
        if isinstance(val, OpsMixin):
            if for_read and not val.can_read:
                raise QCDLUserError(f"can not read {val}")

            value = val.value
            if add_parens and " " in value:
                # don't try to minimize parenthesis for now: all the ops (except
                # negation) will end up with parenthesis
                value = "({value})".format(value=value)
            return value
        else:
            return str(val)

    @staticmethod
    def _assert_compatible_modules(a: Any, b: Any) -> bool | None:
        """We only allow objects with the same QcdlModules to interact with
        each other"""
        mods: list[set[Any]] = [set(), set()]
        for i, obj in enumerate([a, b]):
            if isinstance(obj, _SENS):
                # every qubit has this integer, so it's compatible with
                # everything
                return True
            elif isinstance(obj, (Array, OpsMixin)):
                modules = obj.modules
            elif isinstance(obj, list):
                modules = obj
            else:
                if not isinstance(obj, (int, float)):
                    raise TypeError(
                        f"operations between {a} type {a.__class__}"
                        f" and {b} type {b.__class__} are not supported"
                    )

                # ints, floats are compatible with any set of modules
                return None

            if modules is None:
                # if e.g., SENS was operated on with an integer, neither would
                # have modules
                return True

            mods[i] = set([q.qcdl_module_name for q in modules])

        if mods[0] != mods[1]:
            raise QCDLUserError(f"{a} and {b} do not have compatible modules")

        return None

    def _broadcasted_op(self, op: str, *args: Any) -> RegisterExpression:
        """Apply the operator to construct an RegisterExpression"""
        expr = None
        if op == "__neg__":
            if len(args) != 0:
                raise QCDLInternalError(
                    f"__neg__ is a unary operator and takes no"
                    f" additional arguments, got {args}"
                )
            return RegisterExpression(
                modules=self.modules,
                expr="-{value}".format(value=OpsMixin._get_value(self)),
                master_kwargs=self.master_kwargs,
                scope_id=self.scope_id,
            )

        other = args[0]
        OpsMixin._assert_compatible_modules(self, other)
        is_assignment = op in ["__iadd__", "__isub__", "__imul__", "__ilshift__"]

        lhs = OpsMixin._get_value(self, for_read=not is_assignment)
        # it's an assignment, no additional operators after this one, so no
        # need for more parenthesis
        rhs = OpsMixin._get_value(other, add_parens=not is_assignment)

        if op in ["__rsub__"]:
            lhs, rhs = rhs, lhs

        # if the expression involves SENS, then we may not have modules from
        # these operands
        if getattr(self, "modules", None):
            modules = self.modules
        elif getattr(other, "modules", None):
            modules = other.modules
        else:
            # if it can't find modules from either operand, it can obtain
            # modules from the LHS of the assignment at the end. the case where
            # there is no assignment would require the user to pass the
            # expression to a qcdl statement (where modules isn't needed)
            modules = None

        expr = RegisterExpression(
            modules=modules,
            expr="{lhs} {op} {rhs}".format(
                lhs=lhs,
                op=operators_to_math[op],
                rhs=rhs,
            ),
            master_kwargs=getattr(self, "master_kwargs", None),
            scope_id=self.scope_id,
        )

        if is_assignment:
            expr._emit_instructions()
            return self  # type: ignore[return-value]

        return expr


class AssignmentOpsMixin:
    # _broadcasted_op is provided by OpsMixin in all concrete subclasses.
    _broadcasted_op: Callable[..., Any]

    def __iadd__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__iadd__", other)

    def __isub__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__isub__", other)

    def __imul__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__imul__", other)


class NumberOpsMixin:
    r"""Supported operations for floats.

    Used by the :class:`.FixedPointRegister` class.

    *   :math:`<<=` (assignment)
    *   :math:`==, !=, <, >, <=, >=` (equality)
    *   :math:`+, -, *` (standard arithmetic)
    """
    # _broadcasted_op is provided by OpsMixin in all concrete subclasses.
    _broadcasted_op: Callable[..., Any]

    def __eq__(self, other: object) -> Any:
        return self._broadcasted_op("__eq__", other)

    def __ne__(self, other: object) -> Any:
        return self._broadcasted_op("__ne__", other)

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __lt__(self, other: Any) -> Any:
        return self._broadcasted_op("__lt__", other)

    def __gt__(self, other: Any) -> Any:
        return self._broadcasted_op("__gt__", other)

    def __le__(self, other: Any) -> Any:
        return self._broadcasted_op("__le__", other)

    def __ge__(self, other: Any) -> Any:
        return self._broadcasted_op("__ge__", other)

    def __add__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__add__", other)

    def __radd__(self, other: Any) -> RegisterExpression:
        """`a+b` is indistinguishable from `b+a` but we have to add both to the
        mixin to support `reg + 1` and `1 + reg` in Python. Same story
        for __rmul__
        """
        return self._broadcasted_op("__add__", other)

    def __sub__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__sub__", other)

    def __rsub__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__rsub__", other)

    def __mul__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__mul__", other)

    def __rmul__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__mul__", other)

    def __neg__(self) -> RegisterExpression:
        return self._broadcasted_op("__neg__")

    # dev note: update the docstring when adding new methods

class IntegerOpsMixin(NumberOpsMixin):
    r"""Supported operations for integers.

    Used by the :class:`.Register` class.

    *   :math:`<<=` (assignment)
    *   :math:`==, !=, <, >, <=, >=`
    *   :math:`+, -, *`
    *   :math:`\&, |`, ^ (bitwise operations)
    *   right shift
    """
    def __and__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__and__", other)

    def __rand__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__and__", other)

    def __or__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__or__", other)

    def __ror__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__or__", other)

    def __xor__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__xor__", other)

    def __rxor__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__xor__", other)

    def __rshift__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__rshift__", other)

    # dev note: update the docstring when adding new methods

class Target(OpsMixin, QcdlModuleContainerBase):
    """A Target represents a location on a qubit that may be written to. This
    object should not be directly instantiated by an end user."""

    def __ilshift__(self, other: Any) -> RegisterExpression:
        return self._broadcasted_op("__ilshift__", other)

    def serialize(self) -> Any:  # type: ignore[override]
        return dict(
            type="cpu", category="target", value=self.value, cls=self.__class__.__name__
        )

    @property
    @abc.abstractmethod
    def value(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def modules(self) -> Sequence[QcdlModule]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.value

    @property
    def _is_qcdl_module(self) -> bool:
        return False

    @property
    def qcdl_modules(self) -> Sequence[QcdlModule]:
        # this allows the procedure decorator to rewrap the QcdlModules for the
        # invoked procedure
        return self.modules


class RegisterExpression(IntegerOpsMixin, OpsMixin, QcdlArgument):
    """An RegisterExpression is composed of operations on registers. It should not be
    instantiated by an end user"""

    def __init__(
        self,
        modules: Sequence[QcdlModule],
        expr: str,
        scope_id: int | None = None,
        master_kwargs: dict[str, Any] | None = None,
    ):
        self.modules = modules
        self.value = expr
        self.can_read = True
        self.scope_id = scope_id
        self.master_kwargs = master_kwargs

    def _emit_instructions(self) -> None:
        """Generate the qcdl instruction and execute it on the QcdlModules"""
        q, kwargs = prepare_multiqubit_stmt(self.modules)
        if self.master_kwargs:
            kwargs.update(self.master_kwargs)

        if q.procedure.expression_queue is not None:
            q.procedure.expression_queue.append(str(self.value))
        else:
            q.cpu(str(self.value), scope_id=self.scope_id, **kwargs)

    def serialize(self) -> dict[str, Any]:
        return dict(type="cpu", category="expression", value=self.value)


class Register(IntegerOpsMixin, AssignmentOpsMixin, RegisterInitializerMixin, Target):
    """18-bit signed integer register.

    Supports a range of values between -131,072 and 131,071
    (:math:`[-2^{17}, 2^{17-1}]`)).

    .. note:: The dual-rail simulator in the |cloud|_ service supports
        floating-point numbers for this register.

    The :ref:`qcdl_basic_registers_arithmetic` section introduces registers.
    Use the ``<<=`` (in-place left shift) operator to assign a value to a
    register. See the :ref:`mixin classes <gate_registers_mixins>` for supported
    operations.

    .. admonition:: Aliasing

        You can alias registers, meaning that two registers point to the same
        memory address. This can be useful if, for example, you want to use the
        :class:`.Register` interface to manipulate memory already allocated by
        another register; a specialized use case is using bitwise operators on a
        :class:`.FixedPointRegister`. This is supported with
        `type punning <https://en.wikipedia.org/wiki/Type_punning>`_, as shown
        in the example below.

    Args:
        modules: Modules where this register is created, typically representing
            one or more qubits. Typically, you create a register from a
            :class:`~dwave.gate.qcdl.Scope` object, which handles this parameter
            for you.
        initial_value: Initial value. Defaults to 0.
        name: Name for this register; useful for troubleshooting. If None, a
            name is generated. See the ``alias`` parameter for type punning.
        master_kwargs: Propagate this to the master instruction.  This parameter
            is intended for use by developers of QCDL.
        alias: Set to True if you are aliasing an existing register, and for
            type punning reuse that register's name in the ``name`` parameter.
            Aliased registers are not reinitialized.
        ignore_reallocation: If True, the compiler does not reallocate if
            already allocated (and does not raise an exception).
        scope_id: Identity of the :class:`~dwave.gate.qcdl.Scope` this register
            is derived from. The
            :meth:`~dwave.gate.qcdl.QcdlModuleContainer.Register` method sets
            this value when you create a register from a
            :class:`~dwave.gate.qcdl.Scope` instance.

    Examples:
        This example shows operations between registers.

        .. testcode::

            from dwave.gate.qcdl import qcdl, Scope

            @qcdl(2)
            def register_ops(q0, q1):
                sc = Scope(q0, q1)
                r1 = sc.Register(initial_value=2, name="r1") # name for debugging
                r2 = sc.Register()
                r2 <<= 1                            # set r2 to 1
                r2 <<= 2 * r1                       # set r2 to 2*r1

            qcdl_program = register_ops()

        This example demonstrates
        `type punning <https://en.wikipedia.org/wiki/Type_punning>`_.

        .. testcode::

            from dwave.gate.qcdl import qcdl, Scope

            @qcdl(2)
            def punning(q0, q1):
                sc = Scope(q0, q1)
                fr = sc.FixedPointRegister(initial_value=1, name="fr")
                ir = sc.Register(name="fr", alias=True)
                ir += ir & 4

            qcdl_program = punning()

        This example instantiates a register directly for qubit ``q0`` and sets
        its ``scope_id`` to a :class:`~dwave.gate.qcdl.Scope` that includes that
        qubit.

        .. testcode::

            from dwave.gate.qcdl import qcdl, Register, Scope

            @qcdl(2)
            def direct(q0, q1):
                sc = Scope(q0, q1)
                r1 = Register(q0, initial_value=2, name="r1", scope_id=sc.scope_id)

            qcdl_program = direct()

    See Also:
        :class:`.FixedPointRegister`,
        :meth:`~dwave.gate.qcdl.QcdlModuleContainer.Register`
    """

    def __init__(
        self,
        modules: Sequence[QcdlModule],
        initial_value: int = 0,
        name: str | None = None,
        master_kwargs: dict[str, Any] | None = None,
        alias: bool | str = False,
        ignore_reallocation: bool = False,
        scope_id: int | None = None,
    ) -> None:
        self._register_initialization(
            modules,
            name=name,
            initial_value=initial_value,
            dtype="int",
            alias=alias,
            ignore_reallocation=ignore_reallocation,
            scope_id=scope_id,
        )
        self.master_kwargs = master_kwargs


class FixedPointRegister(
    NumberOpsMixin, AssignmentOpsMixin, RegisterInitializerMixin, Target
):
    """Fixed-point register.

    Supports a range of values between -2, inclusive, to 2, non-inclusive
    (:math:`[-2,2)` or :math:`[-2, 2 - 2^{-16}]`). This is Q2.16 in
    `Q notation <https://en.wikipedia.org/wiki/Q_(number_format)>`_: the
    register's 18 bits use 1 bit for the sign, 1 for a digit to the left of the
    decimal, and 16 bits to the right of the decimal, giving a resolution of
    :math:`2^{-16}`.

    .. note:: The dual-rail simulator in the |cloud|_ service supports
        floating-point numbers for this register.

    The :ref:`qcdl_basic_registers_arithmetic` section introduces registers.
    Use the ``<<=`` (in-place left shift) operator to assign a value to a
    register. See the :ref:`mixin classes <gate_registers_mixins>` for supported
    operations and the :class:`.Register` class on using the ``alias`` argument
    for `type punning <https://en.wikipedia.org/wiki/Type_punning>`_.

    Args:
        modules: Modules where this register is created, typically representing
            one or more qubits. Typically, you create a register from a
            :class:`~dwave.gate.qcdl.Scope` object, which handles this parameter
            for you.
        initial_value: Initial value. Defaults to 0.0.
        name: Name for this register; useful for troubleshooting. If None, a
            name is generated. See the ``alias`` parameter for type punning.
        master_kwargs: Propagate this to the master instruction. This parameter
            is intended for use by developers of QCDL.
        alias: Set to True if you are aliasing an existing register, and for
            type punning reuse that register's name in the ``name`` parameter.
            Aliased registers are not reinitialized.
        ignore_reallocation: If True, the compiler does not reallocate if
            already allocated (and does not raise an exception).
        scope_id: Identity of the :class:`~dwave.gate.qcdl.Scope` this register
            is derived from. The
            :meth:`~dwave.gate.qcdl.QcdlModuleContainer.FixedPointRegister`
            method sets this value when you create a register from a
            :class:`~dwave.gate.qcdl.Scope` instance.

    Examples:
        This is a typical example of instantiating a register from a
        :class:`~dwave.gate.qcdl.Scope` object for the relevant qubits.

        .. JP: without the r0 register, simulator gives a cannot-allocate-memory error

        .. testcode::

            from dwave.gate.qcdl import qcdl, Scope
            from dwave.gate.qcdl.operations import measure

            @qcdl(2)
            def create_fixed_reg(q0, q1):
                sc = Scope(q0, q1)
                r0 = sc.Register(name="r0")
                r1 = sc.FixedPointRegister(name="r1")
                r1 <<= 1.274
                q0.h()
                measure(q0, register=r0)

            qcdl_program = create_fixed_reg()

        See additional examples in the :class:`.Register` class.

    See:
        `Fixed-point arithmetic <https://en.wikipedia.org/wiki/Fixed-point_arithmetic>`_
        and
        `Q number format <https://en.wikipedia.org/wiki/Q_%28number_format%29>`_

    See Also:
        :class:`.Register`,
        :meth:`~dwave.gate.qcdl.QcdlModuleContainer.FixedPointRegister`
    """

    def __init__(
        self,
        modules: Sequence[QcdlModule],
        initial_value: float = 0.0,
        name: str | None = None,
        master_kwargs: dict[str, Any] | None = None,
        alias: bool | str = False,
        ignore_reallocation: bool = False,
        scope_id: int | None = None,
    ) -> None:
        self._register_initialization(
            modules,
            name=name,
            initial_value=initial_value,
            dtype="float",
            alias=alias,
            ignore_reallocation=ignore_reallocation,
            scope_id=scope_id,
        )
        self.master_kwargs = master_kwargs


class Output(Target):
    """Outputs for writing.

    This class is intended for use by developers of QCDL and advanced users.

    You can write data to these outputs but not read from them. For usage, see
    the :meth:`~dwave.gate.qcdl.QcdlModuleContainer.append_table_row` method.

    Args:
        modules: Modules, typically qubits.
        name: Reserve a specific name. Defaults to None.
        category: Category of registers. Defaults to None. Supported values are:

            *   DYN
            *   GOF
        master_kwargs: Propagate this to the master instruction. This parameter
            is intended for use by developers of QCDL.
    """

    # fmt: off
    _outputs = [
        "DYN1", "DYN2", "DYN3",
        "GOF0", "GOF1", "GOF2", "GOF3",
        "SSB",
        "MX00", "MX01", "MX10", "MX11",
        "REC0", "REC1", "REC2", "REC3"
    ]
    # fmt: on

    def __init__(
        self,
        modules: Sequence[QcdlModule],
        name: str | None = None,
        category: str | None = None,
        master_kwargs: dict[str, Any] | None = None,
        scope_id: int | None = None,
    ) -> None:
        if not name and not category:
            raise QCDLUserError("must specify either name or category")
        elif name and category:
            raise QCDLUserError("can not specify both name and category")

        self._released = True
        self.category = category
        if category:
            state = modules[0].procedure.state
            intersection_available = state.available_outputs(modules, category)
            if not intersection_available:
                raise QCDLUserError(
                    f"no output in category {category} is available for all modules"
                )

            name = sorted(intersection_available)[0]
            self._released = False
            for m in modules:
                state.reserve_output(m, name=name)

        if name not in Output._outputs:
            raise QCDLUserError(f"{name} is not a valid output")

        self._modules = to_sequence(modules)
        self._value = name
        self.can_read = False
        self.master_kwargs = master_kwargs
        self.scope_id = scope_id

    @property
    def value(self) -> str:
        return self._value

    @property
    def modules(self) -> Any:
        return self._modules

    def release_output(self) -> None:
        if not self.category:
            raise QCDLInternalError("can not release if no category specified")
        if self._released:
            raise QCDLInternalError(f"can not release {self} more than once")
        for m in self.modules:
            m.procedure.state.release_output(m, self.name)
        self._released = True


class _SENS(IntegerOpsMixin, OpsMixin, QcdlArgument):
    """SENS is a read-only integer on the qubit that provides values read from
    the CPU sensors.

    Almost certainly the first operation you'll want to do with it is to apply a
    mask to retrieve the value of an individual sensor, e.g.,

        reg <<= SENS & yb.CPUSensors.INT

    to read the branch condition.
    """

    def __init__(self) -> None:
        self.can_read = True
        self.value = "SENS"
        self.scope_id = None

    def __str__(self) -> str:
        return "<SENS>"

    def serialize(self) -> str:
        return "SENS"


# it's a singleton
SENS = _SENS()


class Array(RegisterInitializerMixin):
    """An array of contiguous addresses in memory.

    You can use an array to access a register, such as the
    :class:`.FixedPointRegister` and :class:`.Register` class registers, based
    on its index in a list.

    .. warning:: The compiler does not check bounds for your access. Use with
        care.

    Args:
        modules: Modules, typically qubits, associated with the register.
        initial_value: Initial values. An integer initializes a list of zeros of
            that length. A list or NumPy array sets a number of values that
            depends on its length.
        name: Optional name for the register.
        dtype: Type of registers in the array. Defaults to "int". Supported
            values are:

            *   "int": Integer.
            *   "float": Floating point.
        master_kwargs: Propagate this to the master instruction. This parameter
            is intended for use by developers of QCDL.
        alias: Set to True if you are aliasing an existing register, and for
            type punning reuse that register's name  in the ``name`` parameter.
            Aliased registers are not reinitialized.
        ignore_reallocation: If True, the compiler does not reallocate if
            already allocated (and does not raise an exception).

    .. note:: The assignment operator is not ``<<=`` for arrays.

    Examples:
        This example uses an array to access and set register values.

        .. todo:: I have not managed to get a working example of Array

        .. testcode::
            :skipif: True

            import numpy as np
            from dwave.gate.qcdl import qcdl, Scope
            from dwave.gate.qcdl.operations import measure

            @qcdl(2)
            def array_example(q0, q1):
                # Create an array of 10 integers, with initial values 0, 1, 2, 3, ...
                arr = Scope.Array(np.arange(10))
                # Create registers
                sc = Scope(q0, q1)
                r1 = sc.Register(3)
                r2 = scope.Register()
                # r gets the value of the array at the index of the current value of r1
                r <<= arr[r1]
                # Set the value of the array at index r1 to 0
                arr[r1] = 0     # Note the assignment operator is not <<=

            qcdl_program = array_example()

    """

    def __init__(
        self,
        modules: Sequence[QcdlModule],
        initial_value: Any,
        name: str | None = None,
        dtype: str = "int",
        master_kwargs: dict[str, Any] | None = None,
        alias: bool | str = False,
        ignore_reallocation: bool = False,
        signed: bool = True,
        scope_id: int | None = None,
    ) -> None:
        if isinstance(initial_value, int):
            length = initial_value
            initial_value = [0] * length
        else:
            length = None
        self._register_initialization(
            modules,
            name=name,
            initial_value=initial_value,
            length=length,
            dtype=dtype,
            alias=alias,
            ignore_reallocation=ignore_reallocation,
            signed=signed,
            scope_id=scope_id,
        )
        self.master_kwargs = master_kwargs

    def __str__(self) -> str:
        return "<Array {val}>".format(val=self.value)

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: Any) -> RegisterExpression:
        OpsMixin._assert_compatible_modules(self, item)
        expr = RegisterExpression(
            modules=self.modules,
            expr="{lhs}[{idx}]".format(
                lhs=self.value,
                idx=OpsMixin._get_value(item, add_parens=False),
            ),
            master_kwargs=self.master_kwargs,
        )
        return expr

    def __setitem__(self, ofs: Any, other: Any) -> None:
        OpsMixin._assert_compatible_modules(self, other)
        OpsMixin._assert_compatible_modules(self, ofs)
        expr = RegisterExpression(
            modules=self.modules,
            expr="{lhs}[{idx}] = {rhs}".format(
                lhs=self.value,
                idx=OpsMixin._get_value(ofs, add_parens=False),
                rhs=OpsMixin._get_value(other),
            ),
            master_kwargs=self.master_kwargs,
            scope_id=self.scope_id,
        )
        expr._emit_instructions()


class ExpressionAggregator:
    """Combine several expressions into one CPU program.

    This class is intended for use by developers of QCDL and advanced users.

    This class can reduce the number of QCDL instructions. Its primary use
    is for GOF category of QPU registers (see the :class:`.Output` class), where
    the number of instruction set architecture (ISA) instructions must be
    controlled.

    Args:
        modules: Module, typically qubits.
        master_kwargs: Forwarded to the master instruction that is emitted.
            Defaults to None. This parameter is intended for use by developers
            of QCDL.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl, Scope
            from dwave.gate.qcdl.registers import ExpressionAggregator

            @qcdl(2)
            def aggregate_example(q0, q1):
                sc = Scope(q0, q1)
                r0 = sc.Register(name="r0")
                r1 = sc.Register(name="r1")

                with ExpressionAggregator(sc.qcdl_modules):
                    # These two instructions are aggregated into one QCDL instruction
                    r0 <<= 1
                    r1 <<= r0

            qcdl_program = aggregate_example()

    """

    def __init__(
        self,
        modules: Sequence[QcdlModule],
        master_kwargs: dict[str, Any] | None = None,
        scope_id: int | None = None,
    ) -> None:
        self.modules = to_sequence(modules)
        self.master_kwargs = master_kwargs
        self.scope_id = scope_id

    @property
    def procedure(self) -> Procedure:
        return self.modules[0].procedure

    def __enter__(self) -> None:
        # Activate the expression_queue, where cpu statements are collected.
        # This prevents the procedure from collecting any statements.
        self.procedure.expression_queue = []

    def __exit__(self, *excinfo: Any) -> None:
        if excinfo and excinfo[0] is not None:
            return

        expression_queue = self.procedure.expression_queue

        # deactivate the expression_queue
        self.procedure.expression_queue = None

        if expression_queue:
            q, kwargs = prepare_multiqubit_stmt(self.modules)  # type: ignore[arg-type]
            if self.master_kwargs:
                kwargs.update(self.master_kwargs)
            # syntax highlighting is not supported either way in vsix,
            # but multi-line is easier to read
            if True:
                expressions = "\n".join(expression_queue)
                expressions = "\n{exprs}".format(exprs=expressions)
            else:
                expressions = expression_queue

            # we've aggregated all the cpu statements into one cpu statement
            # this will put all the cpu programs into one (i.e., one ISA
            # instruction)
            q.cpu(expressions, scope_id=self.scope_id, **kwargs)


def arbitrary_function(
    modules: Sequence[QcdlModule],
    in_dtype: Type[int] | Type[float],
    out_dtype: Type[int] | Type[float],
    name: str | None = None,
    scope_id: int | None = None,
) -> Callable[
    [Callable[[np.ndarray], Sequence[int | float] | np.ndarray]],
    Callable[..., RegisterExpression],
]:
    """Decorator for creating an interpolation table.

    You can use this decorator to create an arbitrary function, such as a
    trigonometric function.

    Args:
        modules: Modules, typically qubits, for which to create the arbitrary
            function.
        in_dtype: Type of numerical value of the input register. The range of
            supported values depends on the selected register (see the
            :class:`.Register` and :class:`.FixedPointRegister` classes).
        out_dtype: Type of numerical value of the output register. The range of
            supported values depends on the selected register.
        name: Name of the interpolation table. Defaults to None.
        scope_id: Identity of the :class:`~dwave.gate.qcdl.Scope` object the
            arbitrary function is derived from.

    Examples:
        This example creates a trigonometric function.

        .. testcode::

            import numpy as np
            from dwave.gate.qcdl import arbitrary_function, qcdl, Scope
            from dwave.gate.qcdl.operations import h

            @qcdl(2)
            def an_arbitrary_func(q0, q1):

                @arbitrary_function(
                    modules=[q0, q1],
                    in_dtype=float,
                    out_dtype=float,
                    name="my_func"
                )
                def sin_half_x(x):
                    return np.sin(np.pi * x / 2)

                sc = Scope(q0, q1)

                r0 = sc.Register(name="r0")

                r1 = sc.FixedPointRegister(name="r1", initial_value=0.5)
                r2 = sc.FixedPointRegister(name="r2")
                r2 <<= sin_half_x(r1)   # set r2 to sin(pi * r1 / 2) = sin(pi/4)

                h(q0)
                measure(q0, register=r0)

            qcdl_program = an_arbitrary_func()

    """
    if in_dtype not in (int, float):
        raise TypeError(f"in_dtype must be int or float, got {in_dtype}")
    if out_dtype not in (int, float):
        raise TypeError(f"out_dtype must be int or float, got {out_dtype}")
    in_scale = FLOAT_TO_INT if in_dtype is int else 1
    out_scale = INT_TO_FLOAT if out_dtype is int else 1

    def arbitrary_function_decorator(
        func: Callable[[np.ndarray], Sequence[int | float] | np.ndarray],
    ) -> Callable[..., RegisterExpression]:
        if name is not None:
            tag = name
        elif func.__name__ == "<lambda>":
            tag = inspect.getsource(func).strip()
        else:
            tag = None

        tag = tag or func.__name__

        def _foo(x: np.ndarray) -> np.ndarray | str:
            result = func(in_scale * x)
            if isinstance(result, str):
                return result
            return np.asarray(result) * out_scale

        q, kwargs = prepare_multiqubit_stmt(modules)
        q.procedure.state.get_or_add_arbitrary_function(
            q,
            tag=tag,
            foo=_foo,
            dtype=out_dtype,
            scope_id=scope_id,
            **kwargs,
        )

        @wraps(func)
        def wrapper(input_register: Any) -> RegisterExpression:
            OpsMixin._assert_compatible_modules(modules, input_register)
            expr = RegisterExpression(
                modules=modules,
                expr="{name}({inp})".format(
                    name=tag,
                    inp=OpsMixin._get_value(input_register, add_parens=False),
                ),
                scope_id=scope_id,
            )
            return expr

        return wrapper

    return arbitrary_function_decorator
