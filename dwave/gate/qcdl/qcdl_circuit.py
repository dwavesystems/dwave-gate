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

"""The circuit coordinates data across the qubits/procedures"""

from __future__ import annotations

import functools
import inspect
import logging
import numbers
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Literal, TypeAlias, overload, Protocol

import numpy as np

from .base import IndexerMixin
from .components import Procedure, QCDLModule
from .exceptions import QCDLInternalError, QCDLUserError
from .qcdl_models import QCDLProgram, QCDLModuleName, QCDLProcedureDef
from .transformer import print_qcdl
from .utils import is_qubit_or_coupler_name

logger = logging.getLogger(__name__)


class Environment(Protocol):
    """Structural type for environments."""

    def get_modules(self, include_couplers: bool) -> Iterable[Any]:
        ...


class Machine(Protocol):
    """Structural type for machine adapters."""

    environment: Environment

    def get_system(self, name: str) -> QCDLModule:
        ...

    def set_up_systems(self, systems: dict[str, Any], procedure: Procedure) -> None:
        ...

    def clean_up_systems(self, systems: dict[str, Any]) -> None:
        ...


class QCDLCircuit(IndexerMixin):
    """Coordinates and stores data across qubits and procedures.

    Also prepares the data structure for transmission to a compiler.

    The code below shows a simple use case.

    .. code-block:: python

        qcdl = QCDLCircuit(environment)

        # create the objects used for invoking statements
        qmods = qcdl.initialize_modules()

        # call whatever instructions on those QCDLModules
        qmods["q0"].x()
        qmods["q0"].measure()

        # convert this data structure to JSON and pass it to the compiler
        qcdl_model = qcdl.to_model()

    """

    BLOCK_SIZE = 1024

    def __init__(
        self,
        environment: Environment | None = None,
        next_indices: dict[str, int] | None = None,
        validate_non_deterministic_qubits_mid: bool = True,
        validate_non_deterministic_qubits_end: bool = True,
    ):
        """Create a QCDL Circuit object

        An environment is only required for certain operations.

        The most common usage for this class is shown in the @qcdl decorator.

        Args:
            environment (Environment | None, optional): Aqumen Environment.
                Defaults to None.
            next_indices (dict[str, int] | None, optional): Seed values for
                per-kind module index counters.  Defaults to None.
            validate_non_deterministic_qubits_mid (bool, optional): Raise if a
                non-deterministic qubit appears in the middle of the circuit.
                Defaults to True.
            validate_non_deterministic_qubits_end (bool, optional): Raise if a
                non-deterministic qubit appears at the end of the circuit.
                Defaults to True.
        """
        super().__init__(next_indices=next_indices)
        self._environment = environment
        self._all_modules: dict[str, QCDLModule] | None = None

        self._output_pools: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: dict(
                DYN=["DYN1", "DYN2", "DYN3"], GOF=["GOF0", "GOF1", "GOF2", "GOF3"]
            )
        )

        self._main: Procedure | None = None
        self._program: QCDLProcedureDef | None = None
        self._procedures: dict[str, QCDLProcedureDef] = {}
        self._validate_non_deterministic_qubits_mid = (
            validate_non_deterministic_qubits_mid
        )
        self._validate_non_deterministic_qubits_end = (
            validate_non_deterministic_qubits_end
        )
        self._non_deterministic_rtcf: set[str] | None = None

        self._arbitrary_funcs: dict[str, dict[str, Any]] | None = None
        # This code matches the interpolation grid in the control system
        x = np.arange(-2.0, 2.0, 2**-7)
        # It would seem that the correct rotation would use 256, but 255 that
        # matches what the qubit is doing from testing
        self.arbfn_x = np.concatenate((x[255:], x[:255]))

    @property
    def environment(self) -> Environment:
        if self._environment is None:
            raise QCDLUserError(
                "This QCDLCircuit was not initialized with an environment"
            )

        return self._environment

    def initialize_modules(
        self,
        main_name: str = "main",
        **kwargs: Any,
    ) -> dict[str, QCDLModule]:
        """Initialize a dict of QCDL modules.

        Args:
            main_name: Name for the main procedure.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: Can only do
                this once.

        Returns:
            Dictionary of QCDL modules.
        """

        if self._main:
            raise QCDLUserError("Can not create 2 main procedures")

        modules = list(self.environment.get_modules(include_couplers=True))

        if len(modules) == 0:
            raise QCDLUserError("can not initialize a program with no modules")

        p = Procedure(main_name, state=self, **kwargs)

        all_mods = {m.name: QCDLModule(m.name, p) for m in modules}
        self.all_modules = all_mods
        self._main = p
        return all_mods

    @property
    def main(self) -> Procedure:
        if self._main is None:
            self._main = Procedure("main", state=self)
        return self._main

    @property
    def all_modules(self) -> dict[str, QCDLModule] | None:
        """A dict of all QCDL modules in the system.

        These objects are not necessarily ready to be used as-is in a circuit,
        needing to be rewrapped based on the procedure. Use the
        :meth:`~dwave.gate.qcdl.QcdlModule.get_other_qcdl_module` instead of
        accessing this property directly.

        Returns:
            QCDL modules objects.
        """
        return self._all_modules

    @all_modules.setter
    def all_modules(self, all_modules: dict[str, QCDLModule]) -> None:
        """This setter should only be called after all QCDLModules are instantiated
        (so that all may be provided together)
        """
        self._all_modules = all_modules

    def get_or_add_arbitrary_function(
        self,
        qubit: QCDLModule,
        tag: Any,
        foo: Any,
        dtype: Any,
        scope_id: int | None = None,
        desc: Any = None,
        validate: bool = True,
        qubits: Any = None,
    ) -> None:
        """Add an arbitrary function.

        An arbitrary function is a table of values stored on the qubit (for
        domain and range of supported values, see the
        :class:`~dwave.gate.qcdl.registers.FixedPointRegister` class).

        The implementation here allows the table to be defined as follows:

        *   An array of 512 values. This is not currently exposed in QCDL as the
            :meth:`~dwave.gate.qcdl.registers.arbitrary_function` decorator
            takes a callable.
        *   A callable function that takes a NumPy array and returns a NumPy
            array or a string (to be evaluated on-server).

        .. note:: Calling this function directly from QCDL is not supported (it
            is used by the :meth:`~dwave.gate.qcdl.registers.arbitrary_function`
            decorator).

        Args:
            qubit: The qubits associated with this arbitrary function.
            tag: Name of the arbitrary function.
            foo: The mechanism for generating the table.
            dtype: Data type of the output.
            desc: Description for the comment.
            validate: Validate the data for range. Skip if you are not using
                this function outside the supported domain.
            qubits: Other qubits to associate with this arbitrary function.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: Only up to
                8 arbitrary functions are allowed.
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: If a string
                is used for an arbitrary function, must be a function of
                :math:`x`.
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: Pass either
                a list of values or a function.
        """
        addr_space = qubit.qcdl_module_name
        if self._arbitrary_funcs is None:
            self._arbitrary_funcs = {}
        if addr_space not in self._arbitrary_funcs:
            self._arbitrary_funcs[addr_space] = {}

        arbfns = self._arbitrary_funcs[addr_space]
        if tag in arbfns:
            return

        if len(arbfns) == 8:
            raise QCDLUserError(f"{addr_space} already has 8 arbitrary functions")

        rounded = True
        if callable(foo):
            y = foo(self.arbfn_x)
        else:
            # this was added to match what the ISA ArbitraryFunctions supports,
            # however, there are no current use cases for this
            y = np.array(foo)

        if isinstance(y, np.ndarray):
            maxy = np.max(y)
            miny = np.min(y)
            msg = "[{min}, {max}]".format(min=miny, max=maxy)
            desc = desc or " " + msg
            if validate and (miny < -2 or maxy >= 2):
                raise QCDLUserError(
                    f"the range of the arbitrary function {msg} exceeds [-2, 2)"
                )

            if len(y) != 512:
                raise QCDLUserError(f"length of array must be 512, not {len(y)}")

            # Implementation detail: do rounding here or later in the ISA?
            # do rounding here to make the json smaller
            if rounded:
                y = np.round(y * 65536.0)
            y = y.tolist()
        elif isinstance(y, str):
            if "x" not in y:
                # compiler will handle full validation
                raise QCDLUserError("arbitrary function must be a function of x")
            desc = desc or y
        elif not isinstance(y, str):
            raise QCDLUserError(f"unsupported type for foo {foo.__class__.__name__}")

        if not qubits:
            qubits = []
        qubits = [q.qcdl_module_name for q in qubits]
        if qubit.qcdl_module_name not in qubits:
            qubits.append(qubit.qcdl_module_name)

        qubit.comment(
            "Defined {dtype} arbitrary_function {tag} on {qubits}: {desc}".format(
                dtype=dtype.__name__, tag=tag, qubits=", ".join(qubits), desc=desc
            )
        )
        qubit.allocate_arbitrary_function(
            tag,
            y,
            desc=desc,
            dtype=dtype.__name__,
            rounded=rounded,
            scope_id=scope_id,
            qubits=qubits,
        )
        arbfns[tag] = True

    def available_outputs(
        self, qubits: QCDLModule | Sequence[QCDLModule], category: str
    ) -> set[str] | None:
        """Outputs that are available for one or more qubits.

        For a list of qubits, searches for an output that is available for
        all the qubits.

        Args:
            qubits: Spaces to search.
            category: DYN or GOF types of registers.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: If the
                category is not DYN or GOF.

        Returns:
            Which outputs are available.
        """
        qubits = [qubits] if not isinstance(qubits, Sequence) else qubits

        intersection_available = None
        for qubit in qubits:
            space = qubit.qcdl_module_name
            if category not in self._output_pools[space]:
                raise QCDLUserError(f"output category {category} is not supported")

            pool = set(self._output_pools[space][category])
            if intersection_available is None:
                intersection_available = pool
            else:
                intersection_available &= pool

        return intersection_available

    def reserve_output(
        self, qubit: QCDLModule, category: str | None = None, name: str | None = None
    ) -> str | None:
        """Reserve an output.

        A way to ensure that multiple parts of code are not trying to use the
        same output simultaneously. Three DYN and four GOF outputs are
        available.

        Args:
            qubit: Where to reserve the output.
            category: DYN or GOF types of registers.
            name: Reserve a specific output (e.g., DYN1).

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: The output
                is unavailable.

        Returns:
            One of the DYN or GOF outputs.
        """
        if not name and not category:
            raise QCDLUserError("must specify either name or category")
        elif name and category:
            raise QCDLUserError("can not specify both name and category")

        if isinstance(name, str) and len(name) > 3:
            # validated in available_outputs
            category = name[:3]

        space = qubit.qcdl_module_name
        if category is None:
            raise QCDLUserError(
                f"could not determine output category from name {name!r}"
            )
        if not self.available_outputs(qubit, category):
            raise QCDLUserError(f"no {category} outputs are available for {space}")

        pool = self._output_pools[space][category]
        if name:
            if name not in pool:
                raise QCDLUserError(f"{name} is not available, can not be reserved")
            pool.remove(name)
            return name
        else:
            return pool.pop()

    def release_output(self, qubit: QCDLModule, output: str) -> None:
        """Release an output back to the pool of available outputs.

        Args:
            qubit: Where the output was reserved.
            output: The DYNi or GOFi to release.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: If the
                output was not reserved.
        """
        space = qubit.qcdl_module_name
        category = output[:3]
        if (
            space not in self._output_pools
            or category not in self._output_pools[space]
            or output in self._output_pools[space][category]
        ):
            raise QCDLUserError(
                f"Can not release output {output} that wasn't reserved on {space}"
            )

        self._output_pools[space][category].append(output)

    def set_or_check_nondeterministic_modules(
        self,
        modules: Iterable[str | QCDLModule | QCDLModuleName],
        validate: bool = True,
        description: str | None = None,
    ) -> None:
        """Check for and set circuits that desynchronize qubits.

        QCDL supports non-deterministic control flow, as described in the
        :ref:`qcdl_advanced_control_flow` section. Your QCDl program must ensure
        that when any qubits execute a non-deterministic section of a circuit,
        all qubits in the circuit must be kept synchronized.

        The first time this method is called it assigns the set of modules.
        Subsequent calls assert that these sets are the same.

        .. note:: Conditional statements in themselves are not non-deterministic
            and the compiler synchronizes them such that the True and False
            branches execute in the same amount of time.

        .. tip:: Some loops are not actually non-deterministic; you can unroll
            such loops in your Python code.

        """
        module_names: set[str] = set(
            [
                (
                    m
                    if isinstance(m, str)
                    else (
                        m.name if isinstance(m, QCDLModuleName) else m.qcdl_module_name
                    )
                )
                for m in modules
            ]
        )

        if self._non_deterministic_rtcf is None:
            self._non_deterministic_rtcf = module_names
        elif module_names != self._non_deterministic_rtcf:
            additions = ", ".join(sorted(module_names - self._non_deterministic_rtcf))
            subtractions = ", ".join(
                sorted(self._non_deterministic_rtcf - module_names)
            )
            description = f" {description}" if isinstance(description, str) else ""
            msg = (
                f"mismatching non-deterministic qubit sets{description}:"
                f" {additions=}; {subtractions=}"
            )
            if validate:
                raise QCDLUserError(msg)
            else:
                logger.error(msg)

    @property
    def procedures(self) -> dict[str, QCDLProcedureDef]:
        return self._procedures

    def get_procedure(self, procedure_name: str) -> QCDLProcedureDef | None:
        return self.procedures.get(procedure_name)

    def register_procedure(self, procedure: Procedure) -> None:
        """Register a procedure.

        Args:
            procedure: Procedure to register.

        Raises:
            :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: If procedure
                is not unique.
        """
        cur_def = procedure.to_model()

        if procedure.is_main:
            self._program = cur_def
            return

        prev_def = self.get_procedure(procedure.proc_name)
        if prev_def is not None:
            if prev_def.statement_hash != cur_def.statement_hash:
                raise QCDLUserError(
                    f"can not overwrite procedure {procedure.proc_name}"
                    " with a different procedure!"
                )
        self.procedures[procedure.proc_name] = cur_def

    def to_model(self) -> QCDLProgram:
        """Build and return the validated QCDL model for this circuit.

        Returns:
            Validated program model, ready to pass to a compiler or convert to a
            plain dict via the :meth:`~pydantic.BaseModel.model_dump` method
            with ``exclude_unset=True``.
        """
        self.main.end_procedure()
        if self._program is None:
            raise QCDLInternalError("program is None after end_procedure()")

        self.set_or_check_nondeterministic_modules(
            self._program.signature.qubits_used,
            validate=self._validate_non_deterministic_qubits_end,
            description="in qubits used in circuit",
        )

        return QCDLProgram(
            program=self._program,
            procedures=self.procedures,
            next_indices=dict(self._next_indices),
        )


def _get_fspec(f: Any) -> tuple[list[str], str | None]:
    if not (inspect.isfunction(f) or inspect.ismethod(f)):
        # allow users to pass a callable object
        f = f.__call__

    # fspec lets us see whether our callable has specific or arbitrary kwargs
    # https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
    fspec = inspect.getfullargspec(f)
    f_keywords = fspec.varkw

    return fspec.args, f_keywords


def _validate_num_qubits(num_qubits: Any) -> None:
    """Check the ``num_qubits`` argument of the :func:`qcdl` decorator.

    Raises:
        :exception:`~dwave.gate.qcdl.exceptions.QCDLUserError`: If
            ``num_qubits`` could not generate at least one qubit.
    """
    if callable(num_qubits):
        raise QCDLUserError(
            f"the qcdl decorator must be called, so decorate"
            f" {getattr(num_qubits, '__name__', num_qubits)} with @qcdl() or"
            f" @qcdl(num_qubits) rather than with a bare @qcdl"
        )
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, numbers.Integral):
        raise QCDLUserError(
            f"num_qubits must be an integer, not {num_qubits!r} of type"
            f" {type(num_qubits).__name__}"
        )
    if num_qubits < 1:
        raise QCDLUserError(
            f"num_qubits must be at least 1, not {num_qubits}; a program needs"
            f" at least one qubit"
        )


def _unfilled_parameters(
    f: Any, args: Sequence[Any], kwarg_names: Iterable[str]
) -> list[str]:
    """Required parameters of ``f`` that this call would leave unbound.

    Args:
        f: The decorated function.
        args: Positional arguments the caller supplied.
        kwarg_names: Names of the keyword arguments the call will supply.

    Returns:
        Parameter names with no value and no default, in declaration order.
    """
    try:
        parameters = inspect.signature(f).parameters.values()
    except (TypeError, ValueError):
        # a callable we can not introspect; let python report the call itself
        return []

    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    positional = [p for p in parameters if p.kind in positional_kinds]
    consumed = {p.name for p in positional[: len(args)]}
    consumed.update(kwarg_names)

    return [
        p.name
        for p in parameters
        if p.default is inspect.Parameter.empty
        and p.kind in positional_kinds + (inspect.Parameter.KEYWORD_ONLY,)
        and p.name not in consumed
    ]


def _unfilled_parameters_error(
    f_name: str,
    missing: Sequence[str],
    supplied: Iterable[str],
    num_qubits: int | None,
    from_environment: bool,
) -> TypeError:
    """Build the error for an entry point whose parameters were not all filled.

    The decorator injects qubits by *name*, so a mistyped or differently named
    parameter silently receives nothing. Python's own message for that names
    only the parameter, which gives no hint that qubits are involved.
    """
    plural = "" if len(missing) == 1 else "s"
    message = (
        f"{f_name}() missing {len(missing)} required positional argument"
        f"{plural}: {', '.join(repr(name) for name in missing)}."
    )

    if from_environment:
        source = "the environment"
    elif num_qubits is not None:
        source = f"@qcdl({num_qubits})"
    else:
        source = "@qcdl()"

    supplied_names = sorted(supplied, key=lambda name: (len(name), name))
    message += (
        f" The qcdl decorator injects qubits as keyword arguments named q<N>"
        f" (q0, q1, ...), and {source} supplied"
        f" {', '.join(supplied_names) if supplied_names else 'none'}."
    )

    if any(is_qubit_or_coupler_name(name) for name in missing):
        message += (
            " Raise num_qubits to cover the missing qubits, or drop the"
            " parameters for them."
        )
    else:
        matches = "does not match" if len(missing) == 1 else "do not match"
        message += (
            f" Parameter{plural} {', '.join(repr(name) for name in missing)}"
            f" {matches} that pattern, so no qubit was injected: rename to"
            f" q0, q1, ..., add a default value, or pass a value explicitly."
        )

    return TypeError(message)


QCDLV2: TypeAlias = str
"""Display-oriented QCDL string representation returned by @qcdl when
``to_qcdlv2=True``.

This form is intended primarily for visualization, debugging, or compatibility
with older QCDL tooling.
"""

QCDLSource: TypeAlias = Callable[..., None]
"""Type of an undecorated QCDL builder function consumed by @qcdl.

The signature is intentionally broad because @qcdl injects qubit/module
arguments at runtime.
"""

QCDLFunc: TypeAlias = Callable[..., QCDLProgram]
"""Decorated callable returned by ``@qcdl`` when producing a payload.

The resulting function may accept ordinary user parameters, while qubit or
module parameters are supplied by the decorator machinery. Calling it returns a
structured QCDL payload.
"""

QCDLFuncV2: TypeAlias = Callable[..., QCDLV2]
"""Decorated callable returned by ``@qcdl`` when producing a v2 display form.

This variant is mainly useful for inspection or rendering of QCDL in an older
textual format, rather than as the richest program representation for downstream
lowering.
"""


@overload
def qcdl(
    num_qubits: int | None = None,
    environment: Environment | None = None,
    machine: Machine | None = None,
    next_indices: dict[str, int] | None = None,
    to_qcdlv2: Literal[False] = False,
    validate_non_deterministic_qubits_mid: bool = True,
    validate_non_deterministic_qubits_end: bool = True,
) -> Callable[[QCDLSource], QCDLFunc]: ...


@overload
def qcdl(
    num_qubits: int | None = None,
    environment: Environment | None = None,
    machine: Machine | None = None,
    next_indices: dict[str, int] | None = None,
    to_qcdlv2: Literal[True] = True,
    validate_non_deterministic_qubits_mid: bool = True,
    validate_non_deterministic_qubits_end: bool = True,
) -> Callable[[QCDLSource], QCDLFuncV2]: ...


def qcdl(
    num_qubits: int | None = None,
    environment: Environment | None = None,
    machine: Machine | None = None,
    next_indices: dict[str, int] | None = None,
    to_qcdlv2: bool = False,
    validate_non_deterministic_qubits_mid: bool = True,
    validate_non_deterministic_qubits_end: bool = True,
) -> Callable[[QCDLSource], Callable[..., QCDLProgram | QCDLV2]]:
    """Decorator to construct a :ref:`QCDL <qcdl_programming_basic>` program.

    The decorated function returns a
    `Pydantic model <https://pydantic.dev/docs/validation/dev/concepts/models/>`_
    that you can submit to a solver in the |cloud|_ service, as described in the
    :ref:`qcdl_submitting_programs` section.

    Args:
        num_qubits: Number of qubits to generate. If you do not specify a number
            of qubits, infers qubits from the signature of the decorated
            function: any ``q<N>`` arguments, where ``<N>`` is an integer, are
            considered qubits. Generated qubits are passed in to the decorated
            function through keyword arguments, so unless the decorated function
            accepts ``**kwargs``, it must declare a ``q<N>`` parameter for each
            generated qubit.
        environment: Environment. The number of qubits supplied is the full set
            supported by the environment. This parameter is intended for use by
            developers of QCDL.
        machine: If a machine is provided, the machine supplies system instances
            instead of the :class:`~dwave.gate.qcdl.QCDLModule` instance. This
            parameter is intended for use by developers of QCDL.
        next_indices: Values from which to start the circuit's indices. This
            facilitates uniqueness across compiler "visitors".
        to_qcdlv2: If True, returns v2 format. Version v2 can hold less
            information than v3, so this is mostly useful for visualization.
        validate_non_deterministic_qubits_mid: If True, validate
            non-deterministic qubits in mid-circuit statements.
        validate_non_deterministic_qubits_end: If True, validate
            non-deterministic qubits at the end of the circuit.

    Examples:
        The first example specifies the number of qubits (three) in the
        decorator.

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import cx, measure, rx

            @qcdl(3)
            def specify_num_qubits(q0, q1, q2, my_angle=0):
                r0 = q0.FixedPointRegister(name="r0", initial_value=my_angle)
                rx(q0, r0)
                cx(q0, q1)
                cx(q1, q2)
                measure(q0)
                measure(q1)
                measure(q2)

            qcdl_model = specify_num_qubits(my_angle=0.5)

        The next example infers the number of qubits (two) from the ``q0, q1``
        arguments in the decorated function.

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import cx, h, measure

            @qcdl()
            def my_bell_circuit(q0, q1):
                h(q0)
                cx(q0, q1)
                measure(q0)
                measure(q1)

            qcdl_model = my_bell_circuit()

    """

    if num_qubits is not None:
        _validate_num_qubits(num_qubits)

    def decorator(f: QCDLSource) -> Callable[..., QCDLProgram | QCDLV2]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> QCDLProgram | QCDLV2:
            if machine and environment:
                raise QCDLUserError("may not provide both machine and environment")

            if machine:
                _env = machine.environment
            else:
                # make sure python sees environment as from the
                # outer scope
                _env = environment

            qcdl_circuit = QCDLCircuit(
                environment=_env,
                next_indices=next_indices,
                validate_non_deterministic_qubits_mid=validate_non_deterministic_qubits_mid,
                validate_non_deterministic_qubits_end=validate_non_deterministic_qubits_end,
            )

            if not _env:
                main = Procedure("main", state=qcdl_circuit)
                qcdl_circuit._main = main

            f_args, f_keywords = _get_fspec(f)

            if _env:
                module_kwargs = qcdl_circuit.initialize_modules()
            elif num_qubits is not None:
                module_kwargs = {
                    "q" + str(q): qcdl_circuit.main.q(q) for q in range(num_qubits)
                }
            else:
                module_kwargs = {
                    q: qcdl_circuit.main.q(q)
                    for q in f_args
                    if is_qubit_or_coupler_name(q)
                }

            if machine:
                # the machine may promote qubits/couplers so its own types
                module_kwargs = {
                    name: machine.get_system(name) for name in module_kwargs
                }

            # kwargs may include modules/systems the user has already created
            merged_kwargs = module_kwargs | kwargs

            # Qubits reach the decorated function by name, so a parameter whose
            # name is not a q<N> gets nothing. Report that (and any qubit this
            # signature has no room for) before running the function, so the
            # failure names the rule instead of an unbound parameter.
            filled = (
                set(merged_kwargs)
                if f_keywords
                else set(merged_kwargs) & set(f_args)
            )
            missing = _unfilled_parameters(f, args, filled)
            if missing:
                raise _unfilled_parameters_error(
                    getattr(f, "__name__", "circuit"),
                    missing,
                    module_kwargs,
                    num_qubits,
                    from_environment=bool(_env),
                )

            if num_qubits is not None and not _env and not f_keywords:
                dropped = [q for q in module_kwargs if q not in f_args]
                if dropped:
                    raise QCDLUserError(
                        f"@qcdl({num_qubits}) generates"
                        f" {', '.join(module_kwargs)} but"
                        f" {getattr(f, '__name__', 'the decorated function')}()"
                        f" has no parameter for {', '.join(dropped)}, so"
                        f" {'they' if len(dropped) > 1 else 'it'} would be"
                        f" dropped from the program; declare a parameter for"
                        f" every generated qubit, lower num_qubits, or accept"
                        f" **kwargs"
                    )

            if machine:
                # let the machine configure the procedure and any other setup it
                # wants
                machine.set_up_systems(merged_kwargs, qcdl_circuit.main)

            if not f_keywords:
                # if there's no f_keywords, then only include keyword args that
                # match f's arguments
                passed_kwargs = {
                    key: val for key, val in merged_kwargs.items() if key in f_args
                }
            else:
                passed_kwargs = merged_kwargs

            try:
                f(*args, **passed_kwargs)
            finally:
                if machine:
                    machine.clean_up_systems(merged_kwargs)

            qcdl_model = qcdl_circuit.to_model()
            if to_qcdlv2:
                result_v2 = print_qcdl(qcdl_model, to_Display=False)
                if result_v2 is None:
                    raise QCDLInternalError("print_qcdl returned None unexpectedly")
                return result_v2
            else:
                return qcdl_model

        return wrapper

    return decorator
