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
The code here facilitates constructing a circuit (in a json format) from python code.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import numbers
import re
import types
from collections.abc import Mapping, Sequence, Set
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, NamedTuple

import numpy as np

from .base import IndexerMixin, QCDLArgument, QCDLModuleContainerBase
from .constants import MIN_INT_REGISTER_VALUE, NUM_RECS
from .exceptions import QCDLInternalError, QCDLUserError
from .qcdl_models import (
    QCDLModuleName,
    QCDLProcedureDef,
    QCDLSignature,
    QCDLStatement,
)
from .records import RecordOutput
from .registers import (
    Array,
    FixedPointRegister,
    Output,
    Register,
    RegisterExpression,
    arbitrary_function,
)
from .statement import Statement  # noqa: F401  kept for public API
from .utils import map_container, objwalk

if TYPE_CHECKING:
    from .qcdl_circuit import QCDLCircuit

logger = logging.getLogger(__name__)


class StatementToHashEncoder(json.JSONEncoder):
    """Support converting a qcdl statement into something jsonifiable
    for purposes of hashing
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, QCDLStatementBridge):
            return str(obj)
        elif hasattr(obj, "unique_name"):
            return obj.unique_name

        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)


class RegisterAllocation(NamedTuple):
    """One entry in the register names a circuit has allocated.

    Args:
        dtype: ``"int"`` or ``"float"``.
        procedure: Procedure that made the allocation. The name it was made
            under is reported when a later declaration clashes, and the
            identity tells a re-run of that same procedure apart from a
            genuine re-declaration.
    """

    dtype: str
    procedure: Procedure


class Procedure(IndexerMixin):
    """A QCDL procedure.


    Args:
        proc_name: Name of the procedure. To ensure a unique name, the
            :class:`~dwave.gate.qcdl.procedure` decorator incorporates
            arguments or keyword arguments (or a hash of these) to create a
            mangled name.
        state: State of the overall circuit.
        modules: Modules refers to the externally facing method signature if set
            to None, taken from ``modules_used`` (i.e., determines the signature
            based on the statements).
        args: Arguments for the signature. Not used for QCDL.
        kwargs: Keyword arguments for the signature.
        qcdl_operator: Original name of the method before mangling.
        is_main: True for the top-most procedure.
    """

    def __init__(
        self,
        proc_name: str,
        state: QCDLCircuit,
        modules: list[QCDLModule] | None = None,
        args: Sequence[Any] | None = None,
        kwargs: Mapping[Any, Any] | None = None,
        qcdl_operator: str | None = None,
        is_main: bool = True,
    ):
        super().__init__()
        self.proc_name = proc_name
        self.state = state
        self.modules = modules
        self.is_main = is_main
        self.statements: list[QCDLStatement] = []
        self._expression_queue: list[Any] | None = None
        self._child_proc: Procedure | None = None

        # modules_used tracks which modules are actually used which may
        # include ones not in the signature.
        # for example, `q0.swap(q1)` might include coupler c0
        self.modules_used: list[QCDLModuleName] = []

        self.args = args or []
        self.kwargs = kwargs or {}
        self.qcdl_operator = qcdl_operator

        # statements added to this procedure are hashed into this for
        # uniqueness checking
        self._statement_hash = hashlib.md5()

        self._continue_labels: list[str] = []
        self._break_labels: list[str] = []

        # this is used as a stack -- add_statement will raise an error if any
        # qubits in the statement are not listed in the last item in this list.
        self._exclusive_modules: list[set[str]] = []

        self._procedure_ended = False

    def to_model(self) -> QCDLProcedureDef:
        """Encoded version of the python code.

        .. note:: The compiler calls the fields ``qubits`` and ``qubits_used``
            even for couplers.
        """

        if not self._procedure_ended:
            raise QCDLInternalError(
                "can not get serializable data until procedure has ended"
            )

        if self.modules:
            qubits: list[QCDLModuleName] = [
                QCDLModuleName.model_validate(str(q)) for q in self.modules
            ]
        else:
            qubits = list(self.modules_used)
        signature = QCDLSignature(
            qcdl_operator=self.qcdl_operator,
            qubits=qubits,
            qubits_used=list(self.modules_used),
            args=QCDLModule.unwrap(self.args),
            kwargs=QCDLModule.unwrap(self.kwargs),
        )
        return QCDLProcedureDef(
            statements=self.statements,
            statement_hash=self.statement_hash,
            signature=signature,
        )

    @property
    def name(self) -> str:
        return self.proc_name

    @property
    def statement_hash(self) -> str:
        """Hash of current statements.

        Used for comparing the statements of two procedures.

        .. tip:: To ensure that the hash cannot change, end the procedure
            (blocking additional statements) before obtaining its hash.

        Returns:
            Hash of the statements.
        """
        if not self._procedure_ended:
            raise QCDLInternalError(
                "can not get statement hash until procedure has ended"
            )

        return self._statement_hash.hexdigest()

    def begin_procedure(
        self,
        name: str,
        **kwargs: Any,
    ) -> Procedure:
        """Start a child procedure of the current procedure.

        .. note:: Argument validation relies on the execution of the python
            code, so not everything is trapped.

        You are not allowed to add a statement to a caller procedure if the
        callee procedure is not ended.
        """
        self._child_proc = Procedure(
            name,
            state=self.state,
            **kwargs,
            is_main=False,
        )
        if self._child_proc is None:
            raise QCDLInternalError("child procedure was not created")
        return self._child_proc

    def end_procedure(self) -> None:
        """End this procedure and register it."""
        self._procedure_ended = True
        if self._exclusive_modules:
            raise QCDLInternalError(
                f"procedure {self} has an unresolved conditional context"
            )
        self.state.register_procedure(self)

    def register_module_used(self, module_name: str | None) -> None:
        """Track the modules a procedure uses.

        Record which modules this procedure uses, which may be different
        from the call-signature modules.

        This list is only for the given QCDL; for example, ``qa.swap(qb)``
        includes ``qa`` and ``qb`` in its signature, but if those qubits are not
        connected, transpilation may add qubits.

        Args:
            module_name: Name of a module.
        """
        if module_name is None:
            return
        module = QCDLModuleName.model_validate(module_name)
        if module not in self.modules_used:
            self.modules_used.append(module)

    def register_memory_allocation(
        self,
        modules: Sequence[QCDLModule],
        name: str,
        dtype: str,
        allow_existing: bool = False,
        initial_value_specified: bool = False,
    ) -> None:
        """Record a register allocation, rejecting a silent re-declaration.

        The compiler keeps the *first* allocation of a name, so a second
        declaration of the same name on the same module is a no-op: its initial
        value never reaches the qubit. That is almost always a mistake, so it is
        reported here instead.

        Register names are global to the circuit rather than local to a
        procedure, so the record lives on the
        :attr:`~dwave.gate.qcdl.qcdl_circuit.QCDLCircuit.allocated_registers`
        attribute of the state, and a name taken in one procedure clashes with
        the same name in another.

        A procedure body is re-executed on every call while the program is
        being built, but is emitted once, so a declaration reached through a
        later run of the *same* procedure is not a re-declaration and is not
        reported.

        Re-declaring the name is allowed when the caller asked for it, but only
        without an initial value: opting in to the re-declaration says the
        existing memory is wanted, whereas giving a value says the opposite,
        and the compiler would ignore it. This applies only once the name is
        allocated; a first allocation always takes its value, whatever the
        caller opted in to.

        This method is mostly intended for use by developers of QCDL; the
        :class:`~dwave.gate.qcdl.registers.Register` and
        :class:`~dwave.gate.qcdl.registers.FixedPointRegister` classes call it
        for you.

        Args:
            modules: Modules the register is allocated on.
            name: Name of the register.
            dtype: ``"int"`` or ``"float"``.
            allow_existing: If True, an existing allocation of ``name`` is
                accepted as long as no initial value was given. Set by the
                ``alias`` and ``ignore_reallocation`` arguments of a register.
                It has no effect when ``name`` is not already allocated.
            initial_value_specified: Whether the caller gave an initial value
                for this register.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: If ``name``
                is already allocated on one of ``modules`` and either
                ``allow_existing`` is False or an initial value was given.
        """
        for module in modules:
            allocated = self.state.allocated_registers.setdefault(
                module.qcdl_module_name, {}
            )
            previous = allocated.get(name)
            if previous is not None and self._is_rerun_of(previous.procedure):
                previous = None
            if previous is not None and not (
                allow_existing and not initial_value_specified
            ):
                if allow_existing:
                    raise QCDLUserError(
                        f"register {name!r} is already allocated on"
                        f" {module.qcdl_module_name} with dtype"
                        f" {previous.dtype} in procedure"
                        f" {previous.procedure.name}, and the compiler keeps"
                        f" the first allocation, so the initial value given"
                        f" here would never reach the qubit. Re-declaring the"
                        f" name is allowed, but giving it a value is not: drop"
                        f" the initial value."
                    )
                raise QCDLUserError(
                    f"register {name!r} is already allocated on"
                    f" {module.qcdl_module_name} with dtype {previous.dtype} in"
                    f" procedure {previous.procedure.name}; register names are"
                    f" global to the circuit and the compiler keeps the first"
                    f" allocation, so this one would be discarded. Reuse the"
                    f" existing register, pick another name, or redeclare it"
                    f" deliberately with alias=True or ignore_reallocation=True"
                    f" and no initial value."
                )
            allocated[name] = RegisterAllocation(dtype, self)

    def _is_rerun_of(self, other: Procedure) -> bool:
        """Whether ``other`` is an earlier run of the procedure ``self`` is.

        Calling a procedure runs its body again, so a register it declares is
        seen once per call even though the procedure is emitted once. Those
        runs are separate :class:`.Procedure` instances sharing a name, and the
        name is what the rest of the circuit deduplicates on, so matching on it
        here agrees with what ends up in the program.
        """
        return other is not self and other.proc_name == self.proc_name

    @property
    def expression_queue(self) -> list | None:
        """Create an expression queue.

        Expression queues let you combine multiple CPU expressions into
        one QCDL statement (see the
        :class:`~dwave.gate.qcdl.registers.ExpressionAggregator` class).

        If this property is None, it is inactive. Otherwise, it is
        active and managed by the
        :class:`~dwave.gate.qcdl.registers.ExpressionAggregator` class, and
        statements are added to it instead of to the procedure.
        """
        return self._expression_queue

    @expression_queue.setter
    def expression_queue(self, expression_queue: list | None) -> None:
        """Activate or deactivate an expression queue.

        Args:
            expression_queue: Must be an empty list to activate the queue or
                None to deactivate it.
        """
        if expression_queue is not None:
            if self._expression_queue is not None:
                raise QCDLInternalError(
                    "can not overwrite pre-existing expression_queue!"
                )
            if not isinstance(expression_queue, list) or expression_queue:
                raise QCDLInternalError(
                    "must initialize expression_queue to an empty list"
                )
        self._expression_queue = expression_queue

    def add_statement(
        self,
        qubit: str | None,
        op: str,
        args: Sequence[Any] | None,
        kwargs: Mapping[Any, Any] | None,
        caller_qubits: Sequence[str] | None = None,
    ) -> None:
        """Append a statement to this procedure.

        A :func:`~dwave.gate.qcdl.procedure` is merely a list of statements.
        This method appends a statement to that list.

        No validation done here on the statement (any method name, arguments,
        and keyword arguments are accepted).

        Args:
            qubit: To give the appearance of the method invoked on an instance,
                this is its name.
            op: Name of the method invoked.
            args: Arguments to the method.
            kwargs: Keyword arguments to the method.
            caller_qubits: Caller qubits.
        """
        if self._child_proc:
            if not self._child_proc._procedure_ended:
                raise QCDLUserError(
                    f"Can't add statement {op} to procedure {self.name} before"
                    f" child procedure {self._child_proc.name} has ended."
                    " Try passing a Scope to the procedure."
                )
            self._child_proc = None

        if self._procedure_ended:
            raise QCDLInternalError(
                f"can not add more statements to ended procedure {self.name}"
            )

        if self.expression_queue is not None and op not in [
            "comment",
            "allocate_memory",
        ]:
            # Only cpu statements are allowed while an expression_queue is
            # active. However, comment and allocate_memory are allowed as
            # exceptions because they're compiler directives (and order doesn't
            # matter for them).
            raise QCDLUserError(
                f"can not add qcdl instruction {op}"
                " while a cpu expression queue is active"
            )

        args = QCDLModule.unwrap(args)
        kwargs = QCDLModule.unwrap(kwargs)

        stmt = QCDLStatement(
            op=op,
            qubit=QCDLModuleName.model_validate(qubit) if qubit is not None else None,
            args=list(args) if args else [],
            kwargs=dict(kwargs) if kwargs else {},
            caller_qubits=[QCDLModuleName.model_validate(q) for q in caller_qubits]
            if caller_qubits
            else [],
        )

        if not stmt.qubits:
            raise QCDLInternalError(f"statement {op} did not use any modules!")

        # if a conditional context is active, then don't allow statements on
        # modules not involved in the context.
        if self._exclusive_modules:
            allowed_modules = self._exclusive_modules[-1]
            # ignore couplers for this check
            used_qubits = {m.name for m in stmt.qubits if m.is_qubit}
            if not used_qubits.issubset(allowed_modules):
                forbidden = ", ".join(used_qubits - allowed_modules)
                context = ", ".join(sorted(allowed_modules))
                raise QCDLUserError(
                    f"you may not add statement {stmt} using qubits {forbidden}"
                    f" while the conditional context {context} is active"
                )

        # track which modules this statement uses
        for m in stmt.qubits:
            self.register_module_used(m.name)

        self._statement_hash.update(
            json.dumps(
                stmt.model_dump(), sort_keys=True, cls=StatementToHashEncoder
            ).encode("utf-8")
        )
        self.statements.append(stmt)

    def __getattr__(self, procedure_name: str) -> Callable[..., None]:
        """Call one procedure from another

        This is similar to __getattr__ on QCDLModules except that
        it can be invoked from a procedure instead of from a QCDLModule.
        The called procedure must have been ended so that this method
        can determine which qubits are sent to it using its signature.

        Args:
            procedure_name (str): name of the procedure to call

        Raises:
            QCDLInternalError: the called procedure must have been ended

        Returns:
            Callable[..., None]: wrapper
        """

        def f(*args: Any, **kwargs: Any) -> None:
            called = self.state.get_procedure(procedure_name)
            if called is None:
                raise QCDLInternalError(
                    "procedure "
                    + procedure_name
                    + " has not been ended, definition is not available"
                )

            caller_qubits = [q.name for q in called.signature.qubits]
            self.add_statement(
                None, procedure_name, args, kwargs, caller_qubits=caller_qubits
            )

        return f

    def q(self, name: str | int) -> QCDLModule:
        """Dynamically get a qubit :class:`~dwave.gate.qcdl.QCDLModule`.

        Args:
            name: Name of the module.

        Returns:
            Module ready for your instructions.
        """
        if isinstance(name, int):
            name = "q%i" % name
        return QCDLModule(name, self)

    def __str__(self) -> str:
        return "<Procedure %s>" % self.proc_name

    def __repr__(self) -> str:
        return str(self)


class QCDLStatementBridge:
    """Object that tracks if a getattr from QCDLModule ever gets resolved"""

    def __init__(self, proc: Procedure, qubit_name: str, method_name: str):
        self._proc = proc
        self._qubit_name = qubit_name
        self._method_name = method_name

    def __str__(self) -> str:
        return "method %s.%s" % (self._qubit_name, self._method_name)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self._proc.add_statement(self._qubit_name, self._method_name, args, kwargs)


class QCDLModuleContainer(QCDLModuleContainerBase):
    """Base class for the :class:`.QCDLModule` and :class:`.Scope` classes.

    .. important:: This class is not meant to be instantiated directly. Typical
        QCDL programs use the :class:`.Scope` class.

    Defines shared methods such as
    :meth:`~dwave.gate.qcdl.QCDLModuleContainer.If`,
    :meth:`~dwave.gate.qcdl.QCDLModuleContainer.comment`, and
    :meth:`~dwave.gate.qcdl.QCDLModuleContainer.sync`.

    """

    @property
    def _is_qcdl_module(self) -> bool:
        return False

    @property
    def scope_id(self) -> int | None:
        """Identity of the container.

        Examples:
            See example for the :attr:`~dwave.gate.qcdl.Scope.scope_id`
            property.
        """
        return None

    def _If(
        self,
        condition: RegisterExpression | QCDLModule | bool | str | None,
        all_sources_identical: bool = True,
        debug: bool = False,
        _exclusivity: bool = False,
        **kwargs: Any,
    ) -> None:
        if (len(self.qcdl_modules) > 1 or kwargs) and not all_sources_identical:
            # this doesn't affect anything if it's a single qubit conditional so
            # exclude it since it's a distraction.
            kwargs["all_sources_identical"] = all_sources_identical

        if debug:
            kwargs["debug"] = True

        if _exclusivity:
            self.procedure._exclusive_modules.append(
                set([q.qcdl_module_name for q in self.qcdl_modules])
            )
        self._multi_qubit_statement(
            "If",
            condition,
            _impl="qubits",
            _allow_override=False,
            **kwargs,
        )

    def _Else(self, true_goto: str | None = None, **kwargs: Any) -> None:
        if true_goto is not None:
            kwargs["true_goto"] = true_goto
        self._multi_qubit_statement(
            "Else", _impl="qubits", _allow_override=False, **kwargs
        )

    def _Endif(
        self, false_goto: str | None = None, _exclusivity: bool = False, **kwargs: Any
    ) -> None:
        if false_goto is not None:
            kwargs["false_goto"] = false_goto
        if _exclusivity:
            self.procedure._exclusive_modules.pop()
        self._multi_qubit_statement(
            "Endif", _impl="qubits", _allow_override=False, **kwargs
        )

    @contextmanager
    def If(
        self,
        condition: RegisterExpression | QCDLModule | bool | str | None,
        all_sources_identical: bool = True,
        debug: bool = False,
        true_goto: str | None = None,
        false_goto: str | None = None,
        _indentation: int = 3,
        **kwargs: Any,
    ) -> Iterator[Callable[..., Any]]:
        """Conditionally execute an expression.

        All qubits in this :class:`.QCDLModuleContainer` participate in the
        conditional. See the :ref:`qcdl_advanced_conditionals` section for
        information on conditional branching.

        Args:
            condition: Branch condition for the qubits. The
                :ref:`qcdl_advanced_conditionals` section describes the
                supported conditions.
            all_sources_identical: For conditional statements with multiple
                qubits, if the same condition computed on each qubit yields the
                same result, skip broadcasting the message.
            true_goto: Upon completion of a True branch, instead of continuing
                to the statement that by default should execute after the ``If``
                body, control moves to the labeled statement. Used for loop
                control flow.
            false_goto: Upon completion of a False branch, instead of continuing
                to the statement that by default should execute after the ``If``
                body, control moves to the labeled statement. Used for loop
                control flow.

        Examples:
            A simple example with one ``If`` statement:

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import h, measure, x, z

                @qcdl(2)
                def single_if(q0, q1):
                    sc = Scope(q0, q1)
                    h(q0)
                    h(q1)
                    measure(q0)
                    with sc.If(True):
                        x(q1)
                    z(q1)
                    measure(q1)

                qcdl_program = single_if()

            A more complex example with nested conditionals:

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import h, measure

                @qcdl(2)
                def nested_if(q0, q1):
                    sc = Scope(q0, q1)
                    r1 = sc.FixedPointRegister(0.75, name="r1")
                    h(q0)
                    h(q1)
                    measure(q0)
                    with sc.If(True) as Else:
                        r1 <<= -0.75
                        measure(q1)
                        with sc.If(False):
                            r1 += 0.1
                    with Else():
                        r1 += 0.2

                qcdl_program = nested_if()
        """
        self._If(
            condition=condition,
            all_sources_identical=all_sources_identical,
            debug=debug,
            _exclusivity=True,
            **kwargs,
        )
        orig_ctx = self.procedure._exclusive_modules[-1]

        @contextmanager
        def _ElseCtx() -> Iterator[None]:
            stmts = self.procedure.statements
            if len(stmts) < 2 or stmts[-1].op != "Endif" or stmts[-2].op != "Else":
                raise QCDLUserError(
                    "The Else context manager must follow an ``If`` context manager"
                )
            stmts.pop()  # Endif
            stmts.pop()  # Else
            # because the True branch popped off the context, now that we know
            # the user wants a False branch, we need to push it back on
            self.procedure._exclusive_modules.append(orig_ctx)
            self._Else(true_goto=true_goto)
            try:
                self.qcdl_indent(indentation=_indentation)
                yield
                self.qcdl_indent(indentation=-_indentation)
            finally:
                self._Endif(
                    false_goto=false_goto,
                    _exclusivity=True,
                )

        try:
            self.qcdl_indent(indentation=_indentation)
            # don't instantiate it here -- leave open the possibility of users
            # providing kwargs here in future versions.
            yield _ElseCtx
            self.qcdl_indent(indentation=-_indentation)
        finally:
            self._Else(true_goto=true_goto)
            self._Endif(
                false_goto=false_goto,
                _exclusivity=True,
            )

    def get_next_index_in_proc(self, name: str) -> int:
        """Return an index that is unique within the current procedure.

        This method is mostly intended for use by developers of QCDL.

        Args:
            name: Namespace for the index.

        Returns:
            int: A unique index.

        Examples:

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import measure

                @qcdl(1)
                def unique(q0):
                    sc = Scope(q0)
                    print(q0.get_next_index_in_proc("q"))
                    print(q0.get_next_index_in_proc("q"))
                    measure(q0)

                qcdl_program = unique()

            The code above prints two unique indices for ``q``.

            .. testoutput::

                0
                1
        """
        return self.procedure.get_next_index(name)

    @contextmanager
    def _loop(
        self, base_name: str, base_idx: int | None = None
    ) -> Iterator[tuple[str, str]]:
        """Set the Continue and Break labels

        Args:
            base_name (str): Base name for the labels
            base_idx (int | None, optional): Reuse an index, otherwise,
               generate a new one.

        Yields:
            continue, break: Labels for continue and break
        """
        if base_idx is None:
            base_idx = self.get_next_index_in_proc(base_name)
        continue_label = f"{base_name}_{base_idx}_start"
        self.procedure._continue_labels.append(continue_label)
        break_label = f"{base_name}_{base_idx}_exit"
        self.procedure._break_labels.append(break_label)
        self.comment()

        # the labels need to be synced across all qubits
        self.sync()
        # beginning of next iteration
        self.Label(continue_label)
        try:
            yield continue_label, break_label
        finally:
            # outside of the loop
            self.Label(break_label)
            self.procedure._break_labels.pop()
            self.procedure._continue_labels.pop()

    @contextmanager
    def While(
        self,
        condition: RegisterExpression | QCDLModule | bool | str | None = None,
        all_sources_identical: bool = True,
        _base_name: str = "while",
        _base_idx: int | None = None,
    ) -> Iterator[None]:
        """Execute context statements while condition is true.

        All qubits in this :class:`.QCDLModuleContainer` participate in the
        conditional. See the :ref:`qcdl_advanced_conditionals` section for
        information on conditional branching.

        Args:
            condition: Branch condition for the qubits. The
                :ref:`qcdl_advanced_conditionals` section describes the
                supported conditions.
            all_sources_identical: For conditional statements with multiple
                qubits, if the same condition computed on each qubit yields the
                same result, skip broadcasting the message.
            _base_name: Label names. Defaults to "while".

        Examples:

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import h, measure

                @qcdl(2)
                def while_use(q0, q1):
                    sc = Scope(q0)
                    r0 = sc.Register(name="r0")
                    r1 = sc.Register(name="r1")
                    r1 <<= 1
                    with sc.While(condition=r1<3):
                        h(q0)
                        measure(q0, register=r0)
                        r1 += 1

                qcdl_program = while_use()
        """
        with self._loop(_base_name, _base_idx) as (continue_label, _):
            with self.If(
                condition=condition,
                all_sources_identical=all_sources_identical,
                true_goto=continue_label,
            ):
                yield

    @contextmanager
    def DoWhile(
        self,
        condition: RegisterExpression | QCDLModule | bool | str | None = None,
        all_sources_identical: bool = True,
    ) -> Iterator[None]:
        """Execute context statements while condition is true, and at least once.

        All qubits in this :class:`.QCDLModuleContainer` participate in the
        conditional. See the :ref:`qcdl_advanced_conditionals` section for
        information on conditional branching.

        Args:
            condition: Branch condition for the qubits. The
                :ref:`qcdl_advanced_conditionals` section describes the
                supported conditions.
            all_sources_identical: For conditional statements with multiple
                qubits, if the same condition computed on each qubit yields the
                same result, skip broadcasting the message.

        Examples:
            This example stops running once a value of zero is measured for
            ``q0``.

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import h, measure, x

                @qcdl(2)
                def while_use(q0, q1):
                    sc = Scope(q0, q1)
                    with sc.DoWhile(condition=q0):
                        h(q0)
                        measure(q0)
                        x(q1)

                qcdl_program = while_use()
        """
        with self._loop("dowhile") as (continue_label, _):
            self.procedure._exclusive_modules.append(
                set([q.qcdl_module_name for q in self.qcdl_modules])
            )
            try:
                yield
            finally:
                with self.If(
                    condition=condition,
                    all_sources_identical=all_sources_identical,
                    true_goto=continue_label,
                ):
                    pass

            self.procedure._exclusive_modules.pop()

    @contextmanager
    def For(
        self,
        loop_register: Any,
        initial_value: Any,
        condition: Any,
        update: Any,
        all_sources_identical: bool = True,
        _base_name: str = "for",
        _base_idx: int | None = None,
    ) -> Iterator[None]:
        """Loop while a specified condition is true.

        An almost C-style for loop, similar to a :meth:`.While` loop with minor
        enhancement.

        All qubits in this :class:`.QCDLModuleContainer` participate in the
        conditional. See the :ref:`qcdl_advanced_conditionals` section for
        information on conditional branching.

        Args:
            loop_register: Register for the loop.
            initial_value: Assigns an initial value to ``loop_register``.
            condition: Branch condition for the qubits. The
                :ref:`qcdl_advanced_conditionals` section describes the
                supported conditions.
            update: Value by which to increment ``loop_register`` in every
                iteration.
            all_sources_identical: For conditional statements with multiple
                qubits, if the same condition computed on each qubit yields the
                same result, skip broadcasting the message.
            _base_name: Label names. Defaults to "for".

        Examples:
            This example runs the loop twice.

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import h, measure

                @qcdl(1)
                def for_loop(q0):
                    sc = Scope(q0)
                    r0 = sc.Register(name="r0")
                    r1 = sc.Register(name="r1")
                    with sc.For(
                        loop_register=r1, initial_value=1, condition=r1<3, update=1
                    ):
                        h(q0)
                        measure(q0, register=r0)

                qcdl_program = for_loop()

        """
        loop_register <<= initial_value
        with self.While(
            condition,
            all_sources_identical=all_sources_identical,
            _base_name=_base_name,
            _base_idx=_base_idx,
        ):
            try:
                yield
                loop_register += update
            finally:
                pass

    @contextmanager
    def Repeat(
        self, number: int | Register, ascending: bool = False
    ) -> Iterator[Register]:
        """Loop over a fixed number of iterations.

        All qubits in this :class:`.QCDLModuleContainer` participate in the
        loop.

        Args:
            number: Number of repetitions. Supports integer values greater than
                :math:`1`. If a specified register would cause an infinite loop,
                performs zero iterations.
            ascending: Increment the counter if True or decrement if False.

        Returns:
            Register: The counter register.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: Integer
                value less than one, which causes an infinite loop.

        Examples:
            This example repeats three times a branch that inverts a qubit and
            measures it.

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import measure, x

                @qcdl(1)
                def repeat_use(q0):
                    sc = Scope(q0)
                    with sc.Repeat(3, ascending=False):
                        x(q0)
                        measure(q0)

                qcdl_program = repeat_use()

        """
        _base_name = "repeat"
        idx = self.get_next_index_in_proc(_base_name)

        # Setup start value and input validation based on input type
        if isinstance(number, Register):
            start = self.Register(name=f"{_base_name}_{idx}_start")
            if ascending:
                start <<= 0
                invalid_start_cond = start >= number
            else:
                start <<= number
                invalid_start_cond = start <= 0
            # If this would be an infinite loop, move the start point so that
            # instead it will not do any iterations.
            with self.If(invalid_start_cond):
                if ascending:
                    start <<= number
                else:
                    start <<= 0
        elif isinstance(number, numbers.Integral) and not isinstance(number, bool):
            if number < 1:
                raise QCDLUserError(
                    f"if not a Register, number must be an integer"
                    f" greater than zero, not {number=}"
                )

            if ascending:
                start = 0
            else:
                start = number
        else:
            raise QCDLUserError(f"number must be a Register or int, not {type(number)}")

        counter = self.Register(name=f"{_base_name}_{idx}")
        if ascending:
            condition = counter < number
            update = 1

        else:
            condition = counter > 0
            update = -1

        with self.For(
            loop_register=counter,
            initial_value=start,
            condition=condition,
            update=update,
            _base_name=_base_name,
            _base_idx=idx,
        ):
            yield counter

    def Break(self) -> None:
        """Exit current :meth:`.While`, :meth:`.DoWhile`, or :meth:`.For` loop.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: If a
                :meth:`.Break` is encountered outside a loop.

        Examples:

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import h, measure, x

                @qcdl(2)
                def break_example(q0, q1):
                    sc = Scope(q0, q1)
                    r0 = sc.Register(name="r0")
                    h(q0)
                    h(q1)
                    measure(q0)
                    with sc.For(
                        loop_register=r0, initial_value=1, condition=r0<0, update=1
                    ):
                        x(q1)
                        with sc.If(False):
                            sc.Break()
                        measure(q1)


                qcdl_program = break_example()
        """
        if not self.procedure._break_labels:
            raise QCDLUserError("not in loop, can not break")
        self.Goto(self.procedure._break_labels[-1])

    def Continue(self) -> None:
        """Start next iteration of current :meth:`.While`, :meth:`.DoWhile`, or
        :meth:`.For` loop.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: If a
                :meth:`.Continue` is encountered outside a loop.

        Examples:

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import h, measure, x

                @qcdl(2)
                def break_example(q0, q1):
                    sc = Scope(q0, q1)
                    r0 = sc.Register(name="r0")
                    h(q0)
                    h(q1)
                    measure(q0)
                    with sc.For(
                        loop_register=r0, initial_value=1, condition=r0<0, update=1
                    ):
                        x(q1)
                        with sc.If(False):
                            sc.Continue()
                        measure(q1)


                qcdl_program = break_example()
        """
        if not self.procedure._continue_labels:
            raise QCDLUserError("not in loop, can not continue")
        self.Goto(self.procedure._continue_labels[-1])

    def comment(self, message: Any = None) -> None:
        """Insert a comment in the QCDL.

        The comment is attached to a single qubit so it is printed just once.

        Args:
            message: Comment to print. If None, prints a blank line.

        Examples:

            .. testcode::

                from dwave.gate.qcdl import print_qcdl, qcdl
                from dwave.gate.qcdl.operations import measure, x

                @qcdl(2)
                def add_comment(q0, q1):
                    x(q0)
                    q1.comment("This is my comment")
                    measure(q0)

                qcdl_program = add_comment()
                print_qcdl(qcdl_program)

            .. testcode::
                :hide:

                print(print_qcdl(qcdl_program))

            The code above prints the following QCDL.

            .. testoutput::
                :options: +NORMALIZE_WHITESPACE

                begin quantum
                   x([q0], q0)
                   # This is my comment
                   measure([q0], q0, log=True)
                end quantum

        """
        q = self.qcdl_modules[0]
        if self.scope_id is not None:
            kwargs = dict(scope_id=self.scope_id)
        else:
            kwargs = {}

        self.procedure.add_statement(q.qcdl_module_name, "comment", [message], kwargs)

    def sync(self, *args: Any, **kwargs: Any) -> None:
        """Synchronize all qubits.

        Synchronizes qubits in the container along with any passed in as
        arguments.

        Examples:
            The code below ensures that all operations before the ``sync`` are
            completed before any operations after the ``sync`` are started.

            .. testcode::

                from dwave.gate.qcdl import qcdl
                from dwave.gate.qcdl.operations import measure, x

                @qcdl(2)
                def add_sync1(q0, q1):
                    sc = Scope(q0, q1)
                    x(q0)
                    sc.sync()
                    measure(q1)

                qcdl_program = add_sync1()

            The above is equivalent to the following use of ``sync`` on the
            qubit.

            .. testcode::

                from dwave.gate.qcdl import qcdl
                from dwave.gate.qcdl.operations import x

                @qcdl(2)
                def add_sync2(q0, q1):
                    x(q0)
                    q0.sync(q1)
                    x(q1)

                qcdl_program = add_sync2()
        """
        self._multi_qubit_statement(
            "sync", *args, _impl="args", _allow_override=False, **kwargs
        )

    def generate_record(self) -> None:
        self._multi_qubit_statement(
            "master",
            length=24,
            gen_rec=True,
            card_name="all_cards",
            _allow_override=False,
        )

    def cpu(self, expression: str, **kwargs: Any) -> None:
        """Add a CPU statement.

        CPU statements are classical operations that run in parallel to quantum
        operations, for example, operations on registers.

        This parameter is intended for use by developers of QCDL. Do not call
        this method directly, use the :class:`Register` or
        :class:`FixedPointRegister` classes instead.

        Args:
            expression: A CPU expression.

        Examples:
            The example below generates a CPU statement seen in the final line
            of the output.

            .. testcode::

                from dwave.gate.qcdl import qcdl, Register
                from dwave.gate.qcdl.operations import measure, x

                @qcdl(1)
                def cpu_example(q0):
                    x(q0)
                    r1 = Register(q0, name="r1")
                    measure(q0)
                    r1 += 1         # This is a CPU instruction

                qcdl_program = cpu_example()
                print_qcdl(qcdl_program)

            .. testcode::
                :hide:

                print(print_qcdl(qcdl_program))

            The code above prints the following QCDL.

            .. testoutput::
                :options: +NORMALIZE_WHITESPACE

                begin quantum
                    x([q0], q0)
                    q0.allocate_memory("r1", initial_value=0, ...
                    measure([q0], q0, log=True)
                    q0.cpu("r1 += 1", scope_id=None)
                end quantum
        """
        self._multi_qubit_statement("cpu", expression, _allow_override=False, **kwargs)

    def all_to_all(
        self, send: RegisterExpression, reduce_op: str, **kwargs: Any
    ) -> None:
        """Send a 1-bit message from all qubits to all other qubits.

        The :ref:`qcdl_advanced_signals` section describes how your QCDL must
        ensure the information in any qubit's register is
        :ref:`mirrored <qcdl_advanced_registers_mirroring>` to all qubits for a
        :ref:`conditional statement <qcdl_advanced_conditionals>`.

        This method implements the following algorithm:

        1.  Evaluate the same expression for each qubit
        2.  Each qubit broadcasts its bit
        3.  The ``reduce_op`` converts the set of input bits into one output bit
        4.  That bit is then sent back to each qubit
        5.  The computed bit is placed on the branch condition of each qubit
            and is therefore identical for all

        Args:
            send: An expression that evaluates to a Boolean.
            reduce_op: The reduce operator.

        Examples:

            .. testcode::

                from dwave.gate.qcdl import qcdl, Scope
                from dwave.gate.qcdl.operations import h, measure, x

                @qcdl(2)
                def all_to_all_use(q0, q1):
                    sc = Scope(q0, q1)
                    r1 = sc.Register(name="r1")
                    h(q0)
                    # alias=True reuses the memory r1 already allocated, so the
                    # outcome is stored on q0 only rather than mirrored
                    measure(q0, register=q0.Register(name="r1", alias=True))
                    sc.all_to_all(send=r1==1, reduce_op="&")
                    with sc.If(None):
                        x(q1)
                        measure(q1)

                qcdl_program = all_to_all_use()

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: Invalid
                reduction operator.

        See Also:
            Examples in the :ref:`qcdl_advanced_signals` section.
        """
        valid_ops = ["&", "^", "|"]
        if reduce_op not in valid_ops:
            raise QCDLUserError(
                f"reduction operator {reduce_op} is not one of"
                f" the valid ops {', '.join(valid_ops)}"
            )
        # all_to_all treats all qubits the same so it doesn't matter which one
        # is used to generate the instruction
        self._multi_qubit_statement(
            "all_to_all",
            send=send,
            reduce_op=reduce_op,
            _allow_override=False,
            **kwargs,
        )

    def barrier(self, *args: Any, label: str | None = None) -> None:
        """Signal to transpiler a border for combining gates.

        The transpiler does not combine gates across a barrier. See the
        :ref:`qcdl_basic_gates_barrier` section for more information.

        Args:
            label: Label for the barrier.

        Examples:
            The code below prevents the transpiler from combining the two
            sequential Pauli-X gates.

            .. testcode::

                from dwave.gate.qcdl import qcdl
                from dwave.gate.qcdl.operations import barrier, measure, x

                @qcdl(1)
                def set_barrier(q0):
                    x(q0)
                    barrier(q0)
                    x(q0)
                    measure(q0)

                qcdl_program = set_barrier()
        """
        if label is not None:
            barrier_kwargs: dict[str, Any] = dict(label=label)
        else:
            barrier_kwargs = {}
        self._multi_qubit_statement(
            "barrier", *args, _impl="args", _allow_override=False, **barrier_kwargs
        )

    def Return(self) -> None:
        """Return from a procedure.

        This method is intended for use by developers of QCDL.
        """
        self.sync(end=True)

    def Label(self, label: str) -> None:
        """Place a label at this point in the instruction sequence.

        :meth:`.Label` and :meth:`.Goto` in QCDL are assumed to be used
        non-deterministically.

        Args:
            label: Name of the label.

        Examples:
            The code below uses the :meth:`.Goto` method to return to statements
            preceded by the :meth:`.Label` method.

            .. testcode::

                from dwave.gate.qcdl import qcdl
                from dwave.gate.qcdl.operations import h, mced, measure

                @qcdl(1)
                def use_goto(q0):
                    sc = Scope(q0)
                    sc.Label("reset")       # Next line resets the qubit
                    q0.reset()

                    # The next line represents the quantum algorithm
                    h(q0)

                    # This section returns to labeled section upon qubit erasure
                    erased = q0.Register(name="erased")
                    erased <<= 0
                    mced(q0, register=erased)
                    with q0.If(erased == 1):
                        sc.Goto("reset")

                    measure(q0)    # Next section of the quantum algorithm

                qcdl_program = use_goto()
        """
        self.state.set_or_check_nondeterministic_modules(
            self.qcdl_modules,
            validate=self.state._validate_non_deterministic_qubits_mid,
            description=f"at label {label}",
        )
        self._multi_qubit_statement("label", label=label, _allow_override=False)

    def Goto(self, label: str) -> None:
        """Goto a label.

        :meth:`.Label` and :meth:`.Goto` in QCDL are assumed to be used
        non-deterministically.

        Args:
            label: Name of the label.

        Examples:
            See examples in the :meth:`.Label` method.
        """
        self.state.set_or_check_nondeterministic_modules(
            self.qcdl_modules,
            validate=self.state._validate_non_deterministic_qubits_mid,
            description=f"at label {label}",
        )
        self._multi_qubit_statement("goto", label=label, _allow_override=False)

    def _multi_qubit_statement(
        self,
        op: str,
        *args: Any,
        _impl: str = "qubits",
        _type_error_raises: bool = False,
        _allow_override: bool = True,
        **kwargs: Any,
    ) -> None:
        """This is a helper method for executing a given statement on all the contained
        qcdl modules.

        TODO: we should consolidate support fewer variants

        _impl (str): How should the multi-qubit statement be implemented (see
            code)
        _type_error_raises (bool): If True, if the requested _impl isn't
            compatible with the method on the qubit, then raise the TypeError
            exception instead of falling back to add_statement (see code).
        _allow_override (bool): If the instance is not a :class:`QCDLModule`, then this
            would allow subclasses or other implementations to override
            :class:`QCDLModule`'s method.
        """
        q, others = self.qcdl_modules[0], self.qcdl_modules[1:]

        def _stmt(name: Any, args: Any, kwargs: Any) -> None:
            use_add_statement = True
            if (
                _allow_override and hasattr(q, op) and type(q) is not QCDLModule
            ):  # Implicitly check if it is either a System or SystemQCDLModule
                # check if it's possible to call on the qubit directly
                f = getattr(q, op)
                sig = inspect.signature(f)
                try:
                    sig.bind(*args, **kwargs)
                    use_add_statement = False
                except TypeError:
                    if _type_error_raises:
                        raise
                    use_add_statement = True

            if use_add_statement:
                # this is direct and can avoid infinite recursions
                self.procedure.add_statement(name, op, args, kwargs)
            else:
                # we can let the contained objects override methods if they want
                getattr(q, op)(*args, **kwargs)

        # we only tag statements with scope_id if it's emitted as a unified
        # statement. This is to avoid requiring functions like delay from
        # supporting scope_id
        if self.scope_id is not None and _impl != "separate":
            kwargs["scope_id"] = self.scope_id

        # We should probably consolidate...
        if _impl == "qubits":
            # this one adds a qubits= kwarg to the signature
            if others:
                kwargs["qubits"] = others
            _stmt(q.qcdl_module_name, args, kwargs)
        elif _impl == "args":
            # this adds all the qubits as args
            multi_qubit_args = list(others) + list(args)
            _stmt(q.qcdl_module_name, multi_qubit_args, kwargs)
        elif _impl == "separate":
            # don't combine into one statement
            for q in self.qcdl_modules:
                _stmt(q.qcdl_module_name, args, kwargs)
        else:
            raise ValueError(f"unknown multi-qubit {_impl=}")

    def qcdl_indent(self, indentation: int = 3) -> None:
        """Add an instruction in the list of instructions to indicate
        that the subsequent statements, when displayed in qcdlv2, should be
        indented. Use a negative number for dedent.

        :meta private:
        """
        if indentation == 0:
            return
        self._multi_qubit_statement("qcdl_indent", indentation, _allow_override=False)

    def qcdl_dedent(self, indentation: int = -3) -> None:
        self.qcdl_indent(indentation=indentation)

    @property
    def OutputRec(self) -> list[Output]:
        return [
            Output(self.qcdl_modules, "REC%d" % i, scope_id=self.scope_id)
            for i in range(NUM_RECS)
        ]

    def Register(self, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a :class:`~dwave.gate.qcdl.registers.Register` for all
        qubits in this container.

        See:
            :class:`~dwave.gate.qcdl.registers.Register`
        """
        return Register(self.qcdl_modules, *args, scope_id=self.scope_id, **kwargs)  # type: ignore[misc]

    def FixedPointRegister(self, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a :class:`~dwave.gate.qcdl.registers.FixedPointRegister`
        for all qubits in this container.

        See:
            :class:`~dwave.gate.qcdl.registers.FixedPointRegister`
        """
        return FixedPointRegister(
            self.qcdl_modules,
            *args,
            scope_id=self.scope_id,
            **kwargs,  # type: ignore[misc]
        )

    def Array(self, *args: Any, **kwargs: Any) -> Any:
        """Instantiate an :class:`~dwave.gate.qcdl.registers.Array` for all
        qubits in this container.

        See:
            :class:`~dwave.gate.qcdl.registers.Array`
        """
        return Array(self.qcdl_modules, *args, scope_id=self.scope_id, **kwargs)  # type: ignore[misc]

    def arbitrary_function(self, *args: Any, **kwargs: Any) -> Any:
        """Instantiate an arbitrary function for all qubits in this container.

        See:
            :func:`~dwave.gate.qcdl.registers.arbitrary_function`
        """
        return arbitrary_function(
            self.qcdl_modules,
            *args,
            scope_id=self.scope_id,
            **kwargs,  # type: ignore[misc]
        )

    def append_table_row(
        self,
        *args: Any,
        table_name: str | None = None,
        shape: RecordOutput | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return register data.

        Result records are used to return register values for your analysis.

        Registers can be tricky to use, with interpretation dependent on
        run-time conditions. This method provides an abstraction layer that
        facilitates easier interpretation.

        This method supports returning multiple tables; each invocation adds one
        row to one of the tables. Column names in the returned table are:

        *   By default, the column name for each register is the register name.
        *   If you pass the register as a keyword argument, the column name is
            the parameter name.
        *   You may also use a tuple, ``(col_name, register)``, to name the
            column.
        *   If the name of the column is None, data is returned anonymously in
            an array instead of as a dict.

        This method also executes post-processing: If the register is a float,
        an int is recast in the returned table.

        Args:
            table_name: Name of the table to append this row to.
            shape: Metadata describing the data. If not specified, created
                based on the registers.
            *args: Registers or literals that you want returned in this row. You
                can leave unspecified; such rows present as NaN or None in the
                returned table.
            **kwargs: Registers or literals that you want returned in this row.
                You can leave unspecified; such rows present as NaN or None in
                the returned table.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: Specified
                an object that is not a register.

        Returns:
            dict: Description of the table row.

        Examples:
            This example appends dict keys ``q0`` and ``q1`` that each has as
            its value a dict with key ``after_swap`` for which the value is the
            returned measurements.

            .. testcode::

                from dwave.gate.qcdl import qcdl
                from dwave.gate.qcdl.operations import swap

                @qcdl(2)
                def table_row(q0, q1):
                    sc = Scope(q0, q1)
                    r1 = sc.Register(name="r1")
                    swap(q0, q1)
                    measure(q0, register=r1)
                    measure(q1, register=r1)
                    sc.append_table_row(r1, table_name="after_swap")

                qcdl_program = table_row()
        """
        shape_provided = shape is not None
        if not shape_provided:
            shape = RecordOutput()

        if not isinstance(shape, RecordOutput):
            raise QCDLUserError("you must pass a RecordOutput as the shape parameter")

        # We will define a schema associated with this row
        table_row_id = self.state.get_next_index("table_row")

        def run_prog(idx: int, arg: Any) -> None:
            # helper method to generate the cpu instruction for the arg
            if hasattr(arg, "name"):
                prog = arg.name
            elif hasattr(arg, "value"):
                prog = arg.value
            elif isinstance(arg, (int, float)):
                prog = arg
            else:
                raise QCDLUserError(f"unknown arg type {arg} has type {type(arg)}")

            self.cpu(f"REC{idx} <<= {prog}")

        # Record which schema to use to interpret the following result record
        # values. We won't rely on MIN_INT_REGISTER_VALUE, but this avoids
        # accidental associations with the schema.
        self.OutputRec[0] <<= MIN_INT_REGISTER_VALUE + table_row_id  # type: ignore[misc]
        next_idx = 1
        num_gen_recs = 0

        args_list: list[Any] = list(args)
        for k, v in kwargs.items():
            args_list.append((k, v))

        for i, raw_arg in enumerate(args_list):
            if isinstance(raw_arg, tuple):
                name, arg = raw_arg
            else:
                arg = raw_arg
                if hasattr(arg, "name"):
                    name = arg.name
                elif hasattr(arg, "value"):
                    name = arg.value
                else:
                    name = f"res{table_row_id}_arg{i}"

            # actually write to the register
            run_prog(next_idx, arg)

            if not shape_provided:
                if isinstance(arg, (float, FixedPointRegister)):
                    shape.double(name=name)
                else:
                    shape.integer(name=name)

            # we can only write 4 at a time, so write these out
            next_idx += 1
            if next_idx % 4 == 0:
                self.generate_record()
                num_gen_recs += 1
                next_idx = 0

        if len(args) != shape.num_primitives:
            raise QCDLUserError(
                f"size of shape {shape.num_primitives} does not equal"
                f" number of arguments {len(args)}"
            )

        # if we didn't finish a complete row, then flush it here
        if next_idx % 4 != 0:
            self.generate_record()
            num_gen_recs += 1

        # Send the schema to the compiler
        # * The first rec value records the table row id.
        # * The compiler stores this metadata for each table row id.
        # * The length of shape is how many records to read. Each shape item
        #   indicates how to post-process the rec (e.g., should it be converted to a
        #   float or not)
        # * This it will define a row which gets added to the indicated table
        table_row = dict(
            table_name=table_name,
            table_row_id=table_row_id,
            num_gen_recs=num_gen_recs,
            shape=shape.to_list(),
        )

        self.comment(f"table {table_name} has shape {shape.description}")
        self._multi_qubit_statement("register_table_row", **table_row)  # type: ignore[arg-type]
        return shape, table_row


class QCDLModule(QCDLModuleContainer):
    """Wrapper around a :class:`.Procedure` instance.

    .. important:: This class is intended for use by developers of QCDL to
        append instructions to a module, typically a qubit.

    A :class:`QCDLModule` instance is associated with a specific
    :class:`.Procedure` instance. This means that if a procedure is entered, a
    new :class:`.QCDLModule` is instantiated.

    .. important:: This class is intended for use by developers of QCDL to
        append instructions to a module, typically a qubit.

    This class is designed to provide an intuitive and convenient way to call
    the :meth:`~dwave.gate.qcdl.components.Procedure.add_statement` method. It does
    not perform validation and does not raise an error for unknown or invalid
    methods and positional/keyword arguments. Use the
    :mod:`~dwave.gate.qcdl.operations` methods instead, where applicable.

    Args:
        module_name: Name of the module, typically a qubit.
        proc: :class:`.Procedure` instance this :class:`.QCDLModule`
            is in.

    Examples:
        This example shows a typical use of :mod:`~dwave.gate.qcdl.operations`
        methods, which indirectly instantiates a :class:`QCDLModule` instance.

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure

            @qcdl(1)
            def direct_op(q0):
                h(q0)
                print(q0.qcdl_module_name)      # Added for the example output
                measure(q0)

            qcdl_program = direct_op()

        The code above prints the
        :attr:`~dwave.gate.qcdl.QCDLModule.qcdl_module_name` property of the
        :class:`QCDLModule` class.

        .. testoutput::

            q0

        This example demonstrates explicit use of the
        :meth:`~dwave.gate.qcdl.components.Procedure.add_statement` method.

        .. testcode::

            from dwave.gate.qcdl import procedure, qcdl
            from dwave.gate.qcdl.operations import h, measure

            @procedure
            def my_procedure(qa, proc_name="proc1"):
                qa.h()

            @qcdl(1)
            def using_qcdlmodule(q0):
                my_procedure(q0)
                q0.procedure.add_statement(
                    qubit="q0", op="measure", args=None, kwargs=None
                )

            qcdl_program = using_qcdlmodule()

    """

    def __init__(
        self,
        module_name: str,
        proc: Procedure,
    ):

        self._module_name: str = str(module_name)
        self._proc: Procedure = proc

    @property
    def _is_qcdl_module(self) -> bool:
        return True

    @staticmethod
    def from_rewrapping(m: QCDLModule, new_proc: Procedure) -> QCDLModule:
        """Repackage a QCDLModule to be appropriate to the procedure scope.

        Args:
            m (QCDLModule): original QCDLModule
            new_proc (Procedure): The new procedure to wrap it in.

        Returns:
            QCDLModule: rewrapped QCDLModule

        :meta private:
        """
        proc = new_proc or m.procedure
        return QCDLModule(m.qcdl_module_name, proc)

    @property
    def qcdl_modules(self) -> tuple[QcdlModule]:
        """The :class:`~dwave.gate.qcdl.QCDLModule` this
        container holds.

        Examples:
            This is an artificial example; see the example for the
            :attr:`~dwave.gate.qcdl.QCDLModule.name` for comparison.

            .. testcode::

                from dwave.gate.qcdl import qcdl
                from dwave.gate.qcdl.operations import h, measure

                @qcdl(1)
                def qcdl_modules(q0):
                    h(q0)
                    print(q0.qcdl_modules[0].name)      # Added for the example output
                    measure(q0)

                qcdl_modules()

            The code above prints the module name.

            .. testoutput::

                q0
        """
        # use a tuple so that it's not editable
        return (self,)

    @property
    def name(self) -> str:
        """str: Name of this instance.

        Examples:

            .. testcode::

                from dwave.gate.qcdl import qcdl
                from dwave.gate.qcdl.operations import h, measure

                @qcdl(1)
                def name_example(q0):
                    h(q0)
                    print(q0.name)      # Added for the example output
                    measure(q0)

                name_example()

            The code above prints the module name.

            .. testoutput::

                q0
        """
        return self.qcdl_module_name

    @property
    def signal(self) -> str:
        """Signal that other qubits can condition a branch upon.

        See the :ref:`qcdl_advanced_signals` section for a description and
        examples of signals and the :ref:`qcdl_advanced_conditionals` section
        for an introduction to conditioned execution.
        """
        return f"{self.qcdl_module_name}.signal"

    def one_to_all(
        self, destinations: Scope, send: RegisterExpression, **kwargs: Any
    ) -> None:
        """Send a bit from one qubit to a :class:`~dwave.gate.qcdl.Scope` of
        other qubits.

        See the :ref:`qcdl_advanced_signals` section for a description and
        examples of signals.

        Args:
            destinations: The scope of all qubits to send the message to. The
                bit is placed on each qubit's branch condition to be used in a
                conditional statement.
            send: The expression to compute the bit on the sender.

        Raises:
            :exception:`ValueError`: If the module has more than one qubit.
        """
        self._multi_qubit_statement(
            "one_to_all",
            send=send,
            qubits=destinations.qcdl_modules,
            scope_id=destinations.scope_id,
            **kwargs,
        )

    @property
    def qcdl_module_name(self) -> str:
        """Return the name of the module.

        Examples:

            .. testcode::

                from dwave.gate.qcdl import qcdl
                from dwave.gate.qcdl.operations import h, measure

                @qcdl(1)
                def qcdl_module_name(q0):
                    h(q0)
                    print(q0.qcdl_module_name)      # Added for the example output
                    measure(q0)

                qcdl_module_name()

            The code above prints the module name.

            .. testoutput::

                q0
        """
        return self._module_name

    @property
    def procedure(self) -> Procedure:
        """Procedure: Procedure this instance is associated with.

        :meta private:
        """
        return self._proc

    @classmethod
    def find_modules(cls, *args: Any, **kwargs: Any) -> Iterator[QCDLModule]:
        """Find all unique modules from the args and kwargs

        Some statements need to do something for every module, for example,
        if every module needs to have identical if branching.

        Since we are not confining modules to specific variables in the function
        signature, we need to go find them instead. This method
        will recursively search all args and kwargs for any modules.

        Yields:
            QCDLModule: modules

        :meta private:
        """
        parameters = [args, kwargs]
        returned = {}
        for path, a in objwalk(parameters):
            if isinstance(a, cls):
                # make sure each module is only returned once
                if a.qcdl_module_name not in returned:
                    yield a
                returned[a.qcdl_module_name] = a
            elif isinstance(a, QCDLModuleContainerBase):
                for q in a.qcdl_modules:
                    if q.qcdl_module_name not in returned:
                        yield q
                    returned[q.qcdl_module_name] = q

    def get_other_qcdl_module(self, module_name: str) -> QCDLModule:
        """Create an ad hoc :class:`QCDLModule` in the same procedure.

        Applying instructions to this will automatically register it as one of
        the modules used by this procedure.

        .. todo:: I could not get a working example for this method

        Args:
            module_name: Name of the other module.

        Raises:
            :class:`~dwave.gate.qcdl.exceptions.QCDLInternalError`: Other
                modules are not available.
            :class:`~dwave.gate.qcdl.exceptions.QCDLUserError`: Other module is
                not in this procedure.

        Returns:
            :class:`~dwave.gate.qcdl.QCDLModule`: The other module is in this
            same procedure scope.
        """
        if not self.state.all_modules:
            raise QCDLInternalError(
                f"QCDL Modules list has not been provided for {self.procedure}"
            )
        if module_name not in self.state.all_modules:
            raise QCDLUserError(
                f"module {module_name} is not available in procedure {self.procedure}"
            )
        m = self.state.all_modules[module_name]
        return QCDLModule.from_rewrapping(m, new_proc=self.procedure)

    @classmethod
    def unwrap(cls, container: Set | Sequence | Mapping | None) -> Any:
        """Prepare an object to be used as args or kwargs in a qcdl statement
        by the compiler

        This method is similar to deepcopy with the exceptions:
        1. references to "leaves" are either preserved or replaced with
           something that is jsonifiable.
        2. containers are replaced with a list or dict (the jsonifiable
           container types)

        The objects that are replaced are:

        * QCDLModule objects and Systems are converted to their string names
          (i.e., "q0", "q3", etc). Their procedure context is encoded elsewhere
          in the json file and does not need to be preserved here.
        * QCDLArguments are also transformed by their to_dict implementation.
        * as a convenience, this method will handle np types

        NOTE: This code is used in contexts where the output is not required to be
        jsonifiable, e.g., the compiler uses this to convert qcdl code into
        qcdl objects. Contexts where it's a requirement to be jsonifiable will
        need to implement their own encoder.

        Args:
            container (Any): Python container

        Returns:
            Any: a new container populated from the old container

        :meta private:
        """

        new_container: Any = None
        for path, raw_obj in objwalk(container, containers=True, sort_items=False):
            if isinstance(raw_obj, np.ndarray):
                obj = raw_obj.tolist()
            elif isinstance(raw_obj, Mapping):
                obj = {}
            elif isinstance(raw_obj, (Sequence, Set)) and not isinstance(raw_obj, str):
                obj = []
            elif isinstance(raw_obj, cls):  # This only catches QCDL modules!
                obj = raw_obj.qcdl_module_name
            elif isinstance(raw_obj, QCDLArgument):
                obj = raw_obj.serialize()
            else:
                obj = raw_obj

            # This code should not attempt to jsonify everything

            if not path:
                new_container = obj
            else:
                parent = new_container
                for elem in path[:-1]:
                    parent = parent[elem]

                if isinstance(parent, list):
                    parent.append(obj)
                else:
                    parent[path[-1]] = obj

        return new_container

    @classmethod
    def rewrap(cls, args: Any, kwargs: Any, proc: Procedure) -> None:
        """Create QCDLModule wrappers around modules in provided args/kwargs

        A QCDLModule is a module wrapped in a Procedure, so when passing a
        module into a procedure, each module must be rewrapped with the
        Procedure. This method will recursively scan args/kwargs and replace
        every QCDLModule with a new QCDLModule (or for a System, reassign proc)

        Args:
            args (Any): container of parameters
            kwargs (Any): container of parameters
            proc (Procedure): a new procedure

        :meta private:
        """

        def mapper(value: Any) -> Any:
            if isinstance(value, cls):
                return value.from_rewrapping(value, new_proc=proc)
            elif isinstance(value, QCDLModuleContainerBase):
                value.set_procedure(proc)
            return value

        map_container(
            [args, kwargs],
            map_value=mapper,
            map_value_instance_types=(cls, QCDLModuleContainerBase),
        )

    def __getattr__(self, name: str) -> QCDLStatementBridge:
        """The __getattr__ implements a mechanism for appending arbitrary
        instructions to the Procedure.

        There is no validation at this level. This method is simply a
        convenience mechanism for appending an instruction that you know the
        compiler or simulator would accept.

        Args:
            name (str): The name of the instruction.

        Returns:
            QCDLStatementBridge: an instance which may be invoked to add args/kwargs to
            the invocation.

        :meta public:
        """
        # NOTE: this does not validate whether the procedure/method exists or not

        # At this point, the returned object is expected to be evaluated as a callable
        # and not actually stored in the qcdl json. A QCDLStatementBridge can help
        # format an appropriate error message if the user intended to get an element.
        return QCDLStatementBridge(self._proc, self.qcdl_module_name, name)

    def __str__(self) -> str:
        return "<QCDLModule %s/%s>" % (self.procedure.name, self.qcdl_module_name)

    def __repr__(self) -> str:
        return str(self)


class Scope(QCDLModuleContainer):
    """Qubits to be used for quantum operations.

    This subclass of :class:`QCDLModuleContainer` facilitates use of features
    related to real-time control flow.

    Args:
        *qubits: Qubits that are all or a subset of those of the main procedure
            decorated by the :func:`~dwave.gate.qcdl.qcdl` decorator.
        use_scope_id: Create the identity of the scope, the
            :attr:`~dwave.gate.qcdl.Scope.scope_id` attribute.

    Examples:
        This example defines a group of two qubits which it passes into a loop
        of three operations on each.

        .. testcode::

            from dwave.gate.qcdl import qcdl, Scope
            from dwave.gate.qcdl.operations import measure, sx

            @qcdl(2)
            def scope_example(q0, q1):
                sc = Scope(q0, q1)
                with sc.Repeat(3):
                    sx(q0)
                    sx(q1)
                measure(q0)
                measure(q1)

            qcdl_program = scope_example()
    """

    def __init__(self, *qubits: QCDLModuleContainer, use_scope_id: bool = True):
        # Deduplicate by assembling a dict
        qubits_by_name = {
            mod.qcdl_module_name: mod for q in qubits for mod in q.qcdl_modules
        }

        # this needs to be a list so that QCDLModuleContainer can update
        # qubits when a Scope goes in and out of procedures.
        self._qubits = list(qubits_by_name.values())

        if len(self._qubits) == 0:
            raise QCDLUserError("an empty Scope is not allowed")

        proc_names = [q.procedure.name for q in self._qubits]
        if len(set(proc_names)) != 1:
            raise QCDLUserError(
                f"Scope was initialized with qubits not all in the"
                f" same procedure {', '.join(proc_names)}"
            )

        # a unique identifier for all statements that are generated with this
        # Scope
        self._scope_id: int | None = (
            self.state.get_next_index("scope_id") if use_scope_id else None
        )

        # look for an exception
        self.procedure

    @property
    def scope_id(self) -> int | None:
        """Identity of the scope.

        Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl, Scope

            @qcdl(2)
            def scope_id_example(q0, q1):
                sc = Scope(q0, q1, use_scope_id=False)
                print(sc.scope_id)

            qcdl_program = scope_id_example()

        The code above creates a scope wihtout an identifier.

            .. testoutput::

                None
        """
        return self._scope_id

    @property
    def qcdl_modules(self) -> list[QCDLModule]:
        """list[QCDLModule]: The QCDL modules, typically qubits, in this scope.

        Examples:
            This example uses the :attr:`~Scope.qcdl_modules` property to add a
            comment in generated QCDL with a qubit's name. In practice, you
            would likely use the simpler :attr:`~dwave.gate.qcdl.QCDLModule.name`
            property (i.e., ``q0.name``)

            .. testcode::

                from dwave.gate.qcdl import print_qcdl, qcdl, Scope
                from dwave.gate.qcdl.operations import measure, sx

                @qcdl(2)
                def modules_property(q0, q1):
                    sc = Scope(q0, q1)
                    sx(q0)
                    measure(q0)
                    sc.comment(f"measure {sc.qcdl_modules[0].name}")

                qcdl_program = modules_property()
                print_qcdl(qcdl_program)

            .. testcode::
                :hide:

                print(print_qcdl(qcdl_program))

            The code above prints the following QCDL.

            .. testoutput::
                :options: +NORMALIZE_WHITESPACE

                begin quantum
                    sx([q0], q0)
                    measure([q0], q0, log=True)
                    # measure q0
                end quantum
        """
        return self._qubits


def procedure(
    f: Any,
    proc_name: str | None = None,
    use_signature_modules: bool = True,
    validate_reused_procedures: bool = True,
) -> Any:
    """Decorator for creating a QCDL procedure.

    Args:
        f: Decorated function.
        proc_name: Procedure name. Every procedure must have a unique name,
            which is determined from the decorated method's signature. If
            unspecified, generates a "mangled" name.
        use_signature_modules: If True, bases procedure name on module name. If
            False, the procedure is defined by its
            :attr:`~dwave.gate.qcdl.Scope.qcdl_modules` attribute.
        validate_reused_procedures: If False and if the procedure has been seen
            before, as determined by its mangled name, it is automatically
            reused. Where safe to do so, set to False to save some time.

            If arguments change from one invocation of a procedure to the next
            (from the Python metaprogramming stage), the contents of the
            procedure may change. If this is a risk for your procedure, set
            to True.

    Examples:
        This example reuses a defined procedure while switching the qubits given
        as parameters.

        .. testcode::

            from dwave.gate.qcdl import procedure, qcdl, Scope
            from dwave.gate.qcdl.operations import h, measure, swap

            @procedure
            def my_procedure(qa, qb, r):
                h(qa)
                qa.sync(qb)
                qb.h()
                swap(qb, qa)
                measure(qa, register=r)

            @qcdl(2)
            def use_procedure(q0, q1):
                sc = Scope(q0, q1)
                r0 = sc.Register(name="r0")
                r1 = sc.Register(name="r1")
                my_procedure(q0, q1, r0)
                my_procedure(q1, q0, r1)

            qcdl_program = use_procedure()

    """

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        args = list(args)  # type: ignore[assignment]
        kwargs = dict(kwargs)

        # all modules that are passed will go into the procedure
        # so search all arguments for a QCDLModule, any will do
        caller = None

        # we need to capture the invoking signature to preserve
        # uniqueness.
        op_name = f.__name__
        op_key = [op_name]
        modules: list[Any] | None = []
        sig_modules = []

        def _add_module(m: Any, _caller: Any, update_op_key: bool = True) -> Any:
            # modifies sig_modules and op_key from outer scope
            # _caller used to set caller in outer scope
            if isinstance(m, QCDLModuleContainerBase) and not isinstance(m, QCDLModule):
                container_op_key = m.op_key  # type: ignore[attr-defined]
                for q in m.qcdl_modules:
                    _caller = _add_module(
                        q, _caller, update_op_key=not bool(container_op_key)
                    )
                if container_op_key:
                    op_key.append(container_op_key)
                return _caller
            qcdl_module_name = m.qcdl_module_name
            if qcdl_module_name in sig_modules:
                if update_op_key:
                    # even if the module was already known, we include it in the
                    # proc name for the sake of uniqueness.
                    op_key.append(qcdl_module_name)
                return _caller
            # Compiler does not allow double entry
            if _caller is None:
                _caller = m.procedure
                if not isinstance(_caller, Procedure):
                    raise QCDLUserError(
                        f"caller of {op_name} must be a Procedure, not {_caller}"
                    )
            sig_modules.append(qcdl_module_name)
            if update_op_key:
                op_key.append(qcdl_module_name)
            return _caller

        if isinstance(f, types.MethodType):
            wrap_args = [f.__self__] + args  # type: ignore[operator]
        else:
            wrap_args = args

        # recursively find all QCDLModules and Systems
        # add them in the order they appear
        for path, a in objwalk([wrap_args, kwargs]):
            if isinstance(a, (FixedPointRegister, Register)):
                # we want the name of the register in the procedure name
                op_key.append(str(a))
                caller = _add_module(a, caller)
            elif isinstance(a, (QCDLModule, QCDLModuleContainerBase)):
                if path[0] == 1:
                    # if it was passed as a kwarg, then include the name of the
                    # kwarg
                    op_key.append(str("_".join([str(p) for p in path[1:]])))
                caller = _add_module(a, caller)
            else:
                op_key.append(str(a))

        # I couldn't find a better way to grab a hold of the calling
        # function than getting it from a module (which should be fine anyway).
        # maybe python3 would support it?

        if caller is None:
            raise QCDLInternalError(
                f"Procedure {op_name} must have a calling Procedure"
            )

        if proc_name is None:
            # to create a procedure name with appropriate uniqueness

            # NOTE: op1 is essentially a str of the method's signature
            op1 = "_".join(op_key)
            op1 = op1.replace(".", "p")  # to match qcdl syntax restrictions
            # make sure it's a valid proc name just so that it could be written back
            # out to qcdlv2: replace all non-word with _
            op1 = re.sub(r"[^\w]", "_", op1)

            op_key_hash = hashlib.md5()
            op_key_hash.update(op1.encode())
            # NOTE: op2 is a unique string prefixed with some human readability
            op2 = op_name + "_" + op_key_hash.hexdigest()

            # both are unique, but op1 can get extremely long
            op = op1 if len(op1) < len(op2) else op2
        else:
            # use provided proc name
            op = str(proc_name)

        # copied from qcdl_grammar.g
        if not re.match(r"^[a-zA-Z_]+\w*$", op):
            raise QCDLInternalError("invalid procedure name " + op)

        if use_signature_modules:
            modules = sig_modules
        else:
            modules = None

        call_method = True
        if op not in caller.state.procedures or validate_reused_procedures:
            p = caller.begin_procedure(
                op,
                modules=modules,
                args=args,
                kwargs=kwargs,
                qcdl_operator=op_name,
            )
            QCDLModule.rewrap(wrap_args, kwargs, p)

            # NOTE: we can't prevent f from changing args or kwargs. If f drops
            # a QCDLModule from args/kwargs, then rewrapping it will need to
            # happen some other way if necessary
            rtrn = f(*args, **kwargs)

            p.end_procedure()
        else:
            call_method = False
            rtrn = None

        # invoke the newly created procedure
        getattr(caller, op)(*args, **kwargs)

        if call_method:
            # Reset modules to prior procedure
            QCDLModule.rewrap(wrap_args, kwargs, caller)

            # in case f altered args or kwargs by dropping modules, reset the
            # modules we collected too since modules doesn't have the
            # args/kwargs structure, the alterations here are in-place (see
            # rewrap) (rewrapping an object twice with the same caller isn't a
            # problem)
            QCDLModule.rewrap(modules, {}, caller)

        return rtrn

    return wrapper
