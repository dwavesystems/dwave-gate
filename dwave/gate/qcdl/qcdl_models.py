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
.. Keeping out of the documentation until this info can be incorporated with
    some reading order

    Pydantic models for the QCDL compiler-facing data structures.

    This module provides Pydantic-based models for validating, documenting, and
    serializing QCDL payloads.  The models add runtime validation, field-level
    documentation, and structured serialization/deserialization.

    :class:`QCDLStatement` also surfaces a full derived API
    (``modules``, ``qargs``, ``condition``, ``simple_desc``, etc.) directly
    on the model for statement-level logic.

"""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.functional_serializers import SerializerFunctionWrapHandler

from .exceptions import QCDLInternalError
from .qcdl_objects import format_signature
from .utils import is_qubit_name, is_qubit_or_coupler_name


def _get_module_from_arg(arg: Any) -> str | None:
    """Return a qubit/coupler name string if *arg* represents one, else ``None``."""
    if isinstance(arg, dict) and arg.get("type") == "variable":
        arg = arg.get("variable")
    if is_qubit_or_coupler_name(arg):
        return arg
    return None


class QCDLModuleName(BaseModel):
    """A qubit, coupler, or arbitrary-prefix module referenced in a QCDL statement.

    Validates from string names used in raw QCDL dicts (e.g., ``"q0"``,
    ``"c3"``, ``"m1"``, ``"m"`` etc.) and serializes back to that form, so
    that the :meth:`~pydantic.BaseModel.model_dump` method on any parent model
    still produces plain strings for the compiler.

    Args:
        kind:
            *   ``"qubit"`` for ``q`` or ``qN`` names
            *   ``"coupler"`` for ``c`` or ``cN`` names
            *   Raw prefix string for any other pattern (e.g. ``"m"`` for ``m``
                or ``mN``).
        index:
            The non-negative integer embedded in the name, or ``None`` when the
            name carries no numeric suffix (e.g. ``"q"`` returns
            ``index=None``).

    Examples:

        >>> from dwave.gate.qcdl.components import QCDLModuleName
        ...
        >>> QCDLModuleName.model_validate("q2")
        QCDLModuleName('q2')
        >>> QCDLModuleName(kind="coupler", index=5).name
        'c5'
        >>> QCDLModuleName.model_validate("m0")
        QCDLModuleName('m0')
        >>> QCDLModuleName.model_validate("m")
        QCDLModuleName('m')
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    index: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _parse_string(cls, v: Any) -> Any:
        if isinstance(v, str):
            prefix = v.rstrip("0123456789")
            digits = v[len(prefix) :]
            index = int(digits) if digits else None
            if not prefix.isidentifier():
                raise ValueError(f"{v!r} is not a valid module name")
            if prefix == "q" and index is not None:
                return {"kind": "qubit", "index": index}
            if prefix == "c" and index is not None:
                return {"kind": "coupler", "index": index}
            if prefix == "q":
                return {"kind": "qubit", "index": None}
            if prefix == "c":
                return {"kind": "coupler", "index": None}
            return {"kind": prefix, "index": index}
        return v

    @model_validator(mode="after")
    def _validate_kind(self) -> QCDLModuleName:
        if not self.kind.isidentifier():
            raise ValueError(f"kind {self.kind!r} is not a valid identifier")
        return self

    @model_serializer
    def _to_name(self) -> str:
        """Serialize to the plain string name (e.g. ``"q0"``, ``"m"``)."""
        return self.name

    @property
    def prefix(self) -> str:
        """String prefix used in the module name (e.g. ``"q"``, ``"c"``, ``"m"``)."""
        if self.kind == "qubit":
            return "q"
        if self.kind == "coupler":
            return "c"
        return self.kind

    @property
    def name(self) -> str:
        """String form of this module.

        Examples:
            Example strings: ``"q0"``, ``"c3"``, ``"m"``.

        """
        return self.prefix if self.index is None else f"{self.prefix}{self.index}"

    @property
    def is_qubit(self) -> bool:
        """``True`` for qubit modules."""
        return self.kind == "qubit"

    @property
    def is_coupler(self) -> bool:
        """``True`` for coupler modules."""
        return self.kind == "coupler"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"QCDLModuleName({self.name!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, QCDLModuleName):
            return self.kind == other.kind and self.index == other.index
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


class QCDLStatement(BaseModel):
    """`Pydantic <https://pydantic.dev/docs/>`_ model for a single QCDL
    statement.

    Every declared field is optional (defaults are provided for all fields).
    The ``extra="allow"`` configuration preserves compiler-specific keys (e.g.
    ``card_name``) so they round-trip transparently through
    :meth:`~pydantic.BaseModel.model_dump`.

    Set the ``exclude_unset=True`` parameter for the
    :meth:`~pydantic.BaseModel.model_dump` method to reproduce the sparse
    :class:`dict` representation that
    :meth:`~dwave.gate.qcdl.components.Procedure.add_statement` method generates
    (fields that are not explicitly set are omitted).

    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    op: str | None = Field(
        default=None,
        description=(
            "The name of the operation.  This could be a gate like ``h`` or"
            " ``cx``; it could be a (unique) procedure name; it could be a"
            " control flow operation; or just about anything else.  QCDL"
            " itself is agnostic to the list of available operations — that"
            " ultimately depends on what will receive the QCDL."
            "  Must be a valid Python identifier when present."
        ),
    )

    @field_validator("op")
    @classmethod
    def _validate_op(cls, v: str | None) -> str | None:
        if v is not None and not v.isidentifier():
            raise ValueError(f"op {v!r} is not a valid identifier")
        return v

    qubit: QCDLModuleName | None = Field(
        default=None,
        description=(
            "To give the appearance of the operation being invoked on an"
            " entity (such as a qubit) this is the name of that entity.  If"
            " ``None`` the statement is treated as a bare procedure call."
        ),
    )
    args: list[Any] = Field(
        default_factory=list,
        description="Positional arguments to the operation (raw serialization field).",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments to the operation (raw serialization field).",
    )
    caller_qubits: list[QCDLModuleName] = Field(
        default_factory=list,
        description=(
            "For procedure-call statements, the qubits passed to the called"
            " procedure.  Empty for normal gate/op statements.  Most callers"
            " should use :attr:`qubits` instead."
        ),
    )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Emit ``name`` alongside ``qubit`` for backward compatibility with
        servers that still read the ``name`` key."""
        data: dict[str, Any] = handler(self)
        if "qubit" in data:
            data["name"] = data["qubit"]
        return data

    # ------------------------------------------------------------------
    # Computed fields
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def qubits(self) -> list[QCDLModuleName]:
        """All modules (qubits and couplers) referenced by this statement.

        Derived from every source in insertion order, without duplicates:
        the :attr:`qubit` field, ``caller_qubits`` (for procedure calls),
        ``kwargs["qubits"]``, module-like ``args``, and ``op``-specific logic
        for ``If``/``Else``/``Endif``.

        This is the primary API for callers that need to know which modules a
        statement touches.  Always current---recomputed on each access so
        mutations via :meth:`reassign_arg` / :meth:`reassign_kwarg` are
        reflected immediately.
        """
        _modules: list[QCDLModuleName] = []

        def _add(q: str | QCDLModuleName) -> None:
            m = q if isinstance(q, QCDLModuleName) else QCDLModuleName.model_validate(q)
            if m not in _modules:
                _modules.append(m)

        if self.qubit:
            _add(self.qubit)

        if self.is_procedure_call:
            for q in self.caller_qubits:
                _add(q)

        for q in self.kwargs.get("qubits", []):
            _add(q)

        _non_qubit_args: list[Any] = []
        for arg in self.args:
            if self.op not in ["comment", "If", "c_if", "c_if_else"] and (
                module_name := _get_module_from_arg(arg)
            ):
                _add(module_name)
            else:
                _non_qubit_args.append(arg)

        if self.op in ["If", "Else", "Endif"]:
            for k, v in self.kwargs.items():
                if k not in ["condition", "source_qubit", "qubits"] and is_qubit_name(
                    v
                ):
                    _add(v)

        return _modules

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def qubit_name(self) -> str | None:
        """Alias for :attr:`qubit` that returns the plain string module name."""
        return self.qubit.name if self.qubit is not None else None

    @property
    def is_procedure_call(self) -> bool:
        """``True`` when :attr:`qubit` is ``None`` (bare procedure call)."""
        return self.qubit is None

    @property
    def modules(self) -> list[QCDLModuleName]:
        """All modules referenced by this statement.

        Alias for :attr:`qubits`.  See that attribute for the derivation
        order and sources.
        """
        return self.qubits

    @property
    def qargs(self) -> list[int]:
        """Integer qubit indices derived from :attr:`modules`."""
        return [m.index for m in self.modules if m.is_qubit and m.index is not None]

    @property
    def qubit_names(self) -> list[str]:
        """Names of all qubits referenced by this statement, from all sources.

        Unlike the :attr:`qubits` property, which returns
        :class:`~dwave.gate.qcdl.components.QCDLModuleName` objects
        for all modules (including couplers), this property returns plain
        string names for qubits only.

        Equivalent to ``[m.name for m in self.modules if m.is_qubit]``.
        """
        return [m.name for m in self.modules if m.is_qubit]

    @property
    def non_module_args(self) -> list[Any]:
        """Arguments that are not qubit/coupler/module references.

        Equivalent to the positional arguments after stripping out any
        module-name strings (mirrors the filtering that the legacy
        ``Statement._args`` attribute performed).
        """
        if self.op in ["comment", "If", "c_if", "c_if_else"]:
            return list(self.args)
        return [a for a in self.args if _get_module_from_arg(a) is None]

    @property
    def cargs(self) -> None:
        """Always ``None`` (Qiskit-style interface)."""
        return None

    @property
    def condition(self) -> Any:
        """The condition value for conditional operations.

        Valid for ``op`` values ``"If"``, ``"c_if"``, and ``"c_if_else"``.
        Raises :class:`~dwave.gate.qcdl.exceptions.QCDLInternalError` for any
        other ``op``.
        """
        if self.op not in ["If", "c_if", "c_if_else"]:
            raise QCDLInternalError(f"op {self.op!r} doesn't have a condition")

        if self.op in ["c_if", "c_if_else"]:
            condition = self.args[-1]
        elif self.args:
            condition = self.args[0]
        elif self.kwargs and "condition" in self.kwargs:
            condition = self.kwargs["condition"]
        else:
            raise QCDLInternalError(f"statement {self!s} doesn't have a condition")

        if isinstance(condition, dict):
            _cond: Any = condition
            if _cond.get("type") == "variable":
                return _cond.get("variable")
            return _cond.get("value")
        return condition

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def reassign_arg(self, i: int, new_value: Any) -> None:
        """Replace the :math:`i`-th entry in the raw ``args`` list."""
        self.args[i] = new_value

    def reassign_kwarg(self, name: str, new_value: Any) -> None:
        """Set or replace the keyword argument ``name`` in ``kwargs`` argument."""
        self.kwargs[name] = new_value

    def simple_desc(self) -> str:
        """Return a short human-readable description of the statement."""
        op = self.op or ""
        if op == "cpu":
            expr = self.args[0].strip().splitlines()
            return "; ".join(expr)
        elif op == "If":
            return f"If({self.condition})"
        elif op in ["Else", "Endif"]:
            goto = self.kwargs.get("goto")
            if goto:
                return f"{op}(goto={goto})"
            return op
        elif op in ["label", "goto"]:
            label = self.kwargs["label"]
            return f"{op}({label})"
        return str(self)

    def __str__(self) -> str:
        """QCDL text representation of the statement."""
        op = self.op or ""
        # kwargs passed to format_signature should not include the internal
        # "qubits" key (it is represented via modules/qubits args instead).
        display_kwargs = {k: v for k, v in self.kwargs.items() if k != "qubits"}
        if self.is_procedure_call:
            return format_signature(
                op,
                qubits=[str(q) for q in self.modules],
                args=self.args,
                nargs=display_kwargs,
            ).strip()
        return format_signature(
            op,
            qubit=str(self.qubit) if self.qubit else None,
            args=self.args,
            nargs=display_kwargs,
        ).strip()


class QCDLSignature(BaseModel):
    """`Pydantic <https://pydantic.dev/docs/>`_ model for a procedure signature.

    All fields are required. The ``extra="forbid"`` setting in ``model_config``
    rejects unrecognized keys so that signature construction errors surface
    early.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    qcdl_operator: str | None = Field(
        description=(
            "A non-unique human-readable name for a procedure.  Because"
            " procedure names must be globally unique within a :class:`QCDLProgram`"
            " payload, ``qcdl_operator`` provides a stable label that"
            " multiple distinct procedures may share.  For example, every"
            " ``cz`` implementation on each qubit pair will have a unique"
            " procedure name but may all carry ``qcdl_operator='cz'``."
            "  Simulator backends can use this to identify what a procedure"
            " logically implements."
        ),
    )
    qubits: list[QCDLModuleName] = Field(
        description="Qubits declared in the procedure specification.",
    )
    qubits_used: list[QCDLModuleName] = Field(
        description=(
            "Qubits the procedure actually touches after transpilation."
            "  This may include more entries than ``qubits`` when the"
            " coupling map requires ancillary qubits."
        ),
    )
    args: list[Any] = Field(
        description="Positional arguments accepted by the procedure.",
    )
    kwargs: dict[str, Any] = Field(
        description="Keyword arguments accepted by the procedure.",
    )


class QCDLProcedureDef(BaseModel):
    """`Pydantic <https://pydantic.dev/docs/>`_ model for a compiled procedure
    definition.

    Nesting :class:`QCDLStatement` and :class:`QCDLSignature`
    means that calling
    ``QCDLProcedureDef.model_validate(proc_dict)`` recursively
    validates the entire sub-tree.
    """

    model_config = ConfigDict(extra="forbid")

    statements: list[QCDLStatement] = Field(
        description="The ordered sequence of statements that make up this procedure.",
    )
    statement_hash: str | None = Field(
        default=None,
        exclude=True,
        description=(
            "SHA-based fingerprint of all statements.  Used by"
            " ``QCDLCircuit.register_procedure`` to detect whether a"
            " procedure is being re-registered with a different body, which"
            " is an error.  Excluded from serialized output."
        ),
    )
    signature: QCDLSignature = Field(
        description="Qubit and argument specification for calling this procedure.",
    )


class QCDLProgram(BaseModel):
    """`Pydantic <https://pydantic.dev/docs/>`_ model for a QCDL program.

    .. important:: This class is intended for use by developers of QCDL, though
        the interfaces are of broader usage.

    Use the :class:`~dwave.gate.qcdl.qcdl` decorator to generate an instance of
    this class.


    The ``extra="allow"`` setting in ``model_config`` ensures that
    implementation-specific keys of various compiler versions are preserved
    through round trips.
    """

    model_config = ConfigDict(extra="allow")

    program: QCDLProcedureDef = Field(
        description="The main entry point of the program.",
    )
    procedures: dict[str, QCDLProcedureDef] = Field(
        description=(
            "Named reusable procedures.  Keys are unique procedure names"
            " that may be referenced as a statement ``op``."
            "  Procedure names must be globally unique within a payload."
        ),
    )
    next_indices: dict[str, int] = Field(
        description=(
            "Per-namespace monotonic counters used to generate unique"
            " identifiers (labels, axis IDs, etc.) during circuit"
            " construction."
        ),
    )
