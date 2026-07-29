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
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeAlias, overload

import numpy as np

from .base import IndexerMixin
from .components import Procedure, QcdlModule
from .exceptions import QCDLInternalError, QCDLUserError
from .qcdl_models import Qcdl, QcdlModuleName, QcdlProcedureDef
from .transformer import print_qcdl
from .utils import is_qubit_or_coupler_name

if TYPE_CHECKING:
    from aqumen_environment import Environment
    from aqumen_environment.coupler_graph import CouplerGraph
    from aqumen_environment.modules import Module
    from aqumen_hardware_abstraction import Machine

logger = logging.getLogger(__name__)


class QcdlCircuit(IndexerMixin):
    """The circuit coordinates/stores data across the qubits/procedures and
    prepares the data structure for transmission to the compiler.

    A simple use case would be:

        qcdl = QcdlCircuit(environment)

        # create the objects used for invoking statements
        qmods = qcdl.initialize_modules()

        # call whatever instructions on those QcdlModules
        qmods["q0"].x()
        qmods["q0"].measure()

        # convert this data structure to json and pass it to the compiler
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

        An environment is only required for certain operations (e.g.,
        the coupler_graph).

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
        self._all_modules: dict[str, QcdlModule] | None = None

        self._output_pools: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: dict(
                DYN=["DYN1", "DYN2", "DYN3"], GOF=["GOF0", "GOF1", "GOF2", "GOF3"]
            )
        )

        self._main: Procedure | None = None
        self._program: QcdlProcedureDef | None = None
        self._procedures: dict[str, QcdlProcedureDef] = {}
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
                "This QcdlCircuit was not initialized with an environment"
            )

        return self._environment

    @property
    def coupler_graph(self) -> CouplerGraph:
        return self.environment.coupler_graph

    def initialize_modules(
        self,
        main_name: str = "main",
        modules: Sequence[Module] | None = None,
        **kwargs: Any,
    ) -> dict[str, QcdlModule]:
        """Initialize a dict of QcdlModules

        Args:
            main_name (str, optional): Name for the main procedure.
                Defaults to "main"
            modules (Sequence[Module] | None, optional): Environment modules.
                Defaults to using all in the environment.

        Raises:
            QCDLUserError: can only do this once

        Returns:
            dict[str, QcdlModule]: dictionary of QcdlModules
        """

        if self._main:
            raise QCDLUserError("Can not create 2 main procedures")

        if modules is None:
            modules = list(self.environment.get_modules(include_couplers=True))

        if len(modules) == 0:
            raise QCDLUserError("can not initialize a program with no modules")

        p = Procedure(main_name, state=self, **kwargs)

        all_mods = {m.name: QcdlModule(m.name, p) for m in modules}
        self.all_modules = all_mods
        self._main = p
        return all_mods

    @property
    def main(self) -> Procedure:
        if self._main is None:
            self._main = Procedure("main", state=self)
        return self._main

    @property
    def all_modules(self) -> dict[str, QcdlModule] | None:
        """A dict of all the QcdlModules in the system

        These objects are not necessarily ready to be used in a circuit as-is,
        they probably need to be rewrapped based on the procedure, so clients
        should use `get_other_qcdl_module` instead of accessing this directly.

        Returns:
            dict[str, QcdlModule] | None: QcdlModule objects
        """
        return self._all_modules

    @all_modules.setter
    def all_modules(self, all_modules: dict[str, QcdlModule]) -> None:
        """This setter should only be called after all QcdlModules are instantiated
        (so that all may be provided together)
        """
        self._all_modules = all_modules

    def get_or_add_arbitrary_function(
        self,
        qubit: QcdlModule,
        tag: Any,
        foo: Any,
        dtype: Any,
        scope_id: int | None = None,
        desc: Any = None,
        validate: bool = True,
        qubits: Any = None,
    ) -> None:
        """Add an arbitrary function

        An arbitrary function is a table of values stored on the qubit with both
        domain and range [-2, 2) (which is the limit of a FixedPointRegister)

        The implementation here allows the table to be defined as:
        - an array of 512 values (which correspond to the arbfn's domain). This
          this feature is not currently exposed in QCDL as the arbitrary_function
          decorator takes a callable.
        - a callable function which takes a numpy array and returns a numpy
          array or a string (to be evaluated on server)

        NOTE: it is not supported for QCDL authors to call this directly, this
        is only used by the arbitrary_function decorator.

        Args:
            qubit (QcdlModule): the location this arbfn should be stored
            tag (str): Name of the arbitrary function
            foo (callable or list): The mechanism for generating the table
            dtype (int or float): The data type of the output.
            desc (str, optional): Description for the comment. Defaults to None.
            validate (bool): Validate the data for range. It would be reasonable
               to skip this if you won't use this function in the invalid intervals of
               the domain.
            qubits (QcdlModule, optional): other qubits to add this arbfn

        Raises:
            QCDLUserError: only a max of 8 arbitrary functions are allowed
            QCDLUserError: if a string is used for an arbitrary function, must
                be a function of x
            QCDLUserError: pass either a list of values or a function
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
        self, qubits: QcdlModule | Sequence[QcdlModule], category: str
    ) -> set[str] | None:
        """Which outputs are available for a qubit (or list of qubits)

        If it's a list of qubits, it searches for an output that's available for
        all qubits.

        Args:
            qubits (QcdlModule or list[QcdlModule]): spaces to search
            category (str): DYN or GOF

        Raises:
            QCDLUserError: If the category is not DYN or GOF

        Returns:
            set[str]: which outputs are available
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
        self, qubit: QcdlModule, category: str | None = None, name: str | None = None
    ) -> str | None:
        """Reserve an output (DYN or GOF)

        You can request by specific name (e.g., DYN1) or by category (e.g.,
        DYN).

        This provides a way to ensure that multiple parts of code don't use the
        same output at the same time. Only 3 DYN and 4 GOF are available.

        Args:
            qubit (QcdlModule): where to reserve it
            category (str): DYN or GOF
            name (str): reserve a specific output

        Raises:
            QCDLUserError: the output isn't available

        Returns:
            str: one of the DYN or GOF outputs
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

    def release_output(self, qubit: QcdlModule, output: str) -> None:
        """Release an output back to the pool

        Args:
            qubit (QcdlModule): where the output came from
            output (str): the DYNi or GOFi to return

        Raises:
            QCDLUserError: if the output had not been reserved
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
        modules: Iterable[str | QcdlModule | QcdlModuleName],
        validate: bool = True,
        description: str | None = None,
    ) -> None:
        """Check for circuits which will desynchronize their qubits.

        We allow qubits to use non-deterministic control flow (e.g., loops,
        Label, Goto) and you may even nest them. However, we require that if any qubits
        are handled non-deterministically, then all qubits used in the circuit
        must be handled the same way, because otherwise their qubits would
        desynchronize.

        Some loops are not actually non-deterministic and in principle could be
        unrolled at compile time. Unfortunately, this is not currently
        implemented. If you want to unroll your loops, then you'll need to do
        that in your own Python code.

        The first time this method is called it'll assign the set of modules.
        Subsequent calls will assert that the sets are the same.

        NOTE: conditional statements themselves are not non-deterministic (the
        compiler will sync them so that True and False branches take exactly the
        same amount of time).
        """
        module_names: set[str] = set(
            [
                (
                    m
                    if isinstance(m, str)
                    else (
                        m.name if isinstance(m, QcdlModuleName) else m.qcdl_module_name
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
    def procedures(self) -> dict[str, QcdlProcedureDef]:
        return self._procedures

    def get_procedure(self, procedure_name: str) -> QcdlProcedureDef | None:
        return self.procedures.get(procedure_name)

    def register_procedure(self, procedure: Procedure) -> None:
        """Register a procedure

        Args:
            procedure (Procedure): procedure to register

        Raises:
            QCDLUserError: if procedure is not unique
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

    def to_model(self) -> Qcdl:
        """Build and return the validated QCDL model for this circuit.

        Returns:
            Qcdl: the validated program model, ready to pass to a
                compiler or convert to a plain dict via
                ``model.model_dump(exclude_unset=True)``.
        """
        self.main.end_procedure()
        if self._program is None:
            raise QCDLInternalError("program is None after end_procedure()")

        self.set_or_check_nondeterministic_modules(
            self._program.signature.qubits_used,
            validate=self._validate_non_deterministic_qubits_end,
            description="in qubits used in circuit",
        )

        return Qcdl(
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


QcdlV2: TypeAlias = str
"""Display-oriented QCDL string representation returned by @qcdl when
``to_qcdlv2=True``.

This form is intended primarily for visualization, debugging, or compatibility
with older QCDL tooling.
"""

QcdlSource: TypeAlias = Callable[..., None]
"""Type of an undecorated QCDL builder function consumed by @qcdl.

The signature is intentionally broad because @qcdl injects qubit/module
arguments at runtime.
"""

QcdlFunc: TypeAlias = Callable[..., Qcdl]
"""Decorated callable returned by ``@qcdl`` when producing a payload.

The resulting function may accept ordinary user parameters, while qubit or
module parameters are supplied by the decorator machinery. Calling it returns a
structured QCDL payload.
"""

QcdlFuncV2: TypeAlias = Callable[..., QcdlV2]
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
) -> Callable[[QcdlSource], QcdlFunc]: ...


@overload
def qcdl(
    num_qubits: int | None = None,
    environment: Environment | None = None,
    machine: Machine | None = None,
    next_indices: dict[str, int] | None = None,
    to_qcdlv2: Literal[True] = True,
    validate_non_deterministic_qubits_mid: bool = True,
    validate_non_deterministic_qubits_end: bool = True,
) -> Callable[[QcdlSource], QcdlFuncV2]: ...


def qcdl(
    num_qubits: int | None = None,
    environment: Environment | None = None,
    machine: Machine | None = None,
    next_indices: dict[str, int] | None = None,
    to_qcdlv2: bool = False,
    validate_non_deterministic_qubits_mid: bool = True,
    validate_non_deterministic_qubits_end: bool = True,
) -> Callable[[QcdlSource], Callable[..., Qcdl | QcdlV2]]:
    """Decorator to construct a QCDL dict.

    Args:
        num_qubits (int, optional): Number of qubits to generate.
            Defaults to None. If you do not specify a number of qubits, infers
            qubits from any ``q<N>`` arguments, where ``<N>`` is an int, in the
            signature of the decorated function. Generated qubits are passed in
            to the decorate function through keyword arguments.
        environment (Environment, optional): Environment. The number of qubits
            supplied is the full set supported by the environment. Defaults
            to None.
        machine (Machine | None, optional): If a machine is provided, system
            instances from the machine are supplied instead of QCDL module.
            Defaults to None.
        next_indices (dict[str, int] | None, optional): Start the circuit's indices from
            these values. This facilitates uniqueness across "visitors".
            Defaults to None.
        to_qcdlv2 (bool, optional): If True, returns v2 format.
            Version v2 can hold less information than v3, so this is mostly
            useful for visualization.
        validate_non_deterministic_qubits_mid (bool, optional): If True,
            validate non-deterministic qubits in mid-circuit statements.
        validate_non_deterministic_qubits_end (bool, optional): If True,
            validate non-deterministic qubits at the end of the circuit.

    .. JP: Need more info on ``environment`` and ``machine`` parameters

    Examples:
        The first example specifies the number of qubits (three) in the
        decorator.

        .. testcode::

            from dwave.gate.qcdl import qcdl

            @qcdl(3)
            def specify_num_qubits(q0, q1, q2, my_angle=0):
                r0 = q0.FixedPointRegister(name="r0", initial_value=my_angle)
                q0.rx(r0)
                q0.cx(q1)
                q1.cx(q2)
                q0.measure()
                q1.measure()
                q2.measure()

            qcdl_model = specify_num_qubits(my_angle=0.5)

        The next example infers the number of qubits (two) from the ``q0, q1``
        arguments in the decorated function.

        .. testcode::

            from dwave.gate.qcdl import qcdl

            @qcdl()
            def my_bell_circuit(q0, q1):
                q0.h()
                q0.cx(q1)
                q0.measure()
                q1.measure()

            qcdl_model = my_bell_circuit()

    """

    def decorator(f: QcdlSource) -> Callable[..., Qcdl | QcdlV2]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Qcdl | QcdlV2:
            if machine and environment:
                raise QCDLUserError("may not provide both machine and environment")

            if machine:
                _env = machine.environment
            else:
                # make sure python sees environment as from the
                # outer scope
                _env = environment

            qcdl_circuit = QcdlCircuit(
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
