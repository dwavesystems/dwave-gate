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

"""Tests for the Pydantic QCDL schema models in
:mod:`dwave.gate.qcdl.qcdl_models`.

The fixtures mirror the sparse dict format that
:meth:`~dwave.gate.qcdl.components.Procedure.add_statement` and
:meth:`~dwave.gate.qcdl.qcdl_circuit.QcdlCircuit.to_model` actually produce, so
that tests remain grounded in the real serialisation behaviour.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from dwave.gate.qcdl import procedure, qcdl
from dwave.gate.qcdl.exceptions import QCDLInternalError
from dwave.gate.qcdl.qcdl_models import (
    Qcdl,
    QcdlModuleName,
    QcdlProcedureDef,
    QcdlSignature,
    QcdlStatement,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal valid raw dicts mirroring QcdlCircuit.to_model() output
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_signature() -> dict:
    return {
        "qcdl_operator": None,
        "qubits": ["q0"],
        "qubits_used": ["q0"],
        "args": [],
        "kwargs": {},
    }


@pytest.fixture
def minimal_statement() -> dict:
    """A simple h-gate statement as add_statement would produce it.

    Note: args/kwargs/qubits are present here for fixture completeness, but
    add_statement only includes them when non-empty.
    """
    return {
        "op": "h",
        "qubit": "q0",
        "args": [],
        "kwargs": {},
        "qubits": ["q0"],
        "caller_qubits": [],
    }


@pytest.fixture
def sparse_statement() -> dict:
    """Statement with only op and qubit — as add_statement emits for a plain
    gate with no args, kwargs, or additional qubit references."""
    return {"op": "h", "qubit": "q0"}


@pytest.fixture
def minimal_procedure_def(minimal_statement, minimal_signature) -> dict:
    return {
        "statements": [minimal_statement],
        "statement_hash": "abc123",
        "signature": minimal_signature,
    }


@pytest.fixture
def minimal_qcdl(minimal_procedure_def) -> dict:
    return {
        "program": minimal_procedure_def,
        "procedures": {},
        "next_indices": {},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_compat_name(obj: Any) -> Any:
    """Recursively remove the backward-compat ``"name"`` key that
    ``QcdlStatement`` emits alongside ``"qubit"`` for server
    compatibility.  Used in round-trip tests so that source fixtures
    (which only contain ``"qubit"``) compare equal to dumped output."""
    if isinstance(obj, list):
        return [_strip_compat_name(item) for item in obj]
    if isinstance(obj, dict):
        return {
            k: _strip_compat_name(v)
            for k, v in obj.items()
            if not (k == "name" and "qubit" in obj)
        }
    return obj


# ---------------------------------------------------------------------------
# QcdlModuleName
# ---------------------------------------------------------------------------


class TestQcdlModuleName:
    def test_validate_qubit_from_string(self):
        m = QcdlModuleName.model_validate("q0")
        assert m.kind == "qubit"
        assert m.index == 0
        assert m.name == "q0"

    def test_validate_coupler_from_string(self):
        m = QcdlModuleName.model_validate("c3")
        assert m.kind == "coupler"
        assert m.index == 3
        assert m.name == "c3"

    def test_validate_from_dict(self):
        m = QcdlModuleName.model_validate({"kind": "qubit", "index": 7})
        assert m.name == "q7"

    def test_invalid_name_raises(self):
        with pytest.raises(Exception):
            QcdlModuleName.model_validate("0q")  # digit-leading not allowed
        with pytest.raises(Exception):
            QcdlModuleName.model_validate("q-1")  # non-alphanumeric not allowed

    def test_mixed_case_prefix_no_index(self):
        m = QcdlModuleName.model_validate("qubitA")
        assert m.kind == "qubitA"
        assert m.index is None
        assert m.name == "qubitA"

    def test_mixed_case_prefix_with_index(self):
        m = QcdlModuleName.model_validate("qubitA0")
        assert m.kind == "qubitA"
        assert m.index == 0
        assert m.name == "qubitA0"

    def test_case_insensitive_kind_dict(self):
        m = QcdlModuleName.model_validate({"kind": "QubitA", "index": None})
        assert m.kind == "QubitA"
        assert m.name == "QubitA"

    def test_is_qubit_true(self):
        assert QcdlModuleName.model_validate("q1").is_qubit

    def test_is_coupler_true(self):
        assert QcdlModuleName.model_validate("c0").is_coupler

    def test_str_returns_name(self):
        assert str(QcdlModuleName.model_validate("q5")) == "q5"

    def test_eq_with_model(self):
        a = QcdlModuleName.model_validate("q2")
        b = QcdlModuleName.model_validate("q2")
        assert a == b

    def test_eq_with_string(self):
        m = QcdlModuleName.model_validate("q2")
        assert m == "q2"
        assert m != "q3"

    def test_hash_matches_string_hash(self):
        m = QcdlModuleName.model_validate("q0")
        assert hash(m) == hash("q0")

    def test_serialises_to_string_via_model_dump(self):
        """model_dump() on a parent must still produce plain strings."""
        stmt = QcdlStatement.model_validate({"op": "cx", "qubit": "q0", "args": ["q1"]})
        dumped = stmt.model_dump()
        assert dumped["qubits"] == ["q0", "q1"]

    def test_frozen(self):
        m = QcdlModuleName.model_validate("q0")
        with pytest.raises(Exception):
            m.index = 99  # type: ignore[misc]

    def test_arbitrary_prefix_from_string(self):
        m = QcdlModuleName.model_validate("m0")
        assert m.kind == "m"
        assert m.prefix == "m"
        assert m.index == 0
        assert m.name == "m0"
        assert not m.is_qubit
        assert not m.is_coupler

    def test_arbitrary_prefix_multi_char(self):
        m = QcdlModuleName.model_validate("dr3")
        assert m.kind == "dr"
        assert m.prefix == "dr"
        assert m.index == 3
        assert m.name == "dr3"

    def test_arbitrary_prefix_from_dict(self):
        m = QcdlModuleName.model_validate({"kind": "m", "index": 1})
        assert m.name == "m1"

    def test_arbitrary_prefix_round_trips_via_serializer(self):
        """model_dump() must serialize arbitrary-prefix modules to plain strings."""
        stmt = QcdlStatement.model_validate(
            {"op": "sync", "qubit": "m0", "kwargs": {"qubits": ["m0", "m1"]}}
        )
        dumped = stmt.model_dump()
        assert dumped["qubit"] == "m0"
        assert dumped["qubits"] == ["m0", "m1"]

    def test_prefix_property_for_qubit_and_coupler(self):
        assert QcdlModuleName.model_validate("q4").prefix == "q"
        assert QcdlModuleName.model_validate("c2").prefix == "c"

    def test_no_index_qubit(self):
        m = QcdlModuleName.model_validate("q")
        assert m.kind == "qubit"
        assert m.index is None
        assert m.name == "q"
        assert m.is_qubit

    def test_no_index_coupler(self):
        m = QcdlModuleName.model_validate("c")
        assert m.kind == "coupler"
        assert m.index is None
        assert m.name == "c"
        assert m.is_coupler

    def test_no_index_arbitrary_prefix(self):
        m = QcdlModuleName.model_validate("m")
        assert m.kind == "m"
        assert m.index is None
        assert m.name == "m"

    def test_no_index_from_dict(self):
        m = QcdlModuleName.model_validate({"kind": "m", "index": None})
        assert m.name == "m"

    def test_no_index_serialises_to_string(self):
        stmt = QcdlStatement.model_validate(
            {"op": "h", "qubit": "q", "kwargs": {"qubits": ["q", "m"]}}
        )
        dumped = stmt.model_dump()
        assert dumped["qubit"] == "q"
        assert dumped["qubits"] == ["q", "m"]


# ---------------------------------------------------------------------------
# QcdlStatement
# ---------------------------------------------------------------------------


class TestQcdlStatement:
    def test_all_fields_present(self, minimal_statement):
        stmt = QcdlStatement.model_validate(minimal_statement)
        assert stmt.op == "h"
        assert stmt.qubit == "q0"
        assert stmt.args == []
        assert stmt.kwargs == {}
        # caller_qubits is explicitly set (empty for gate stmts not involving procedure modules)
        assert stmt.caller_qubits == []
        # qubits is the computed set of all referenced modules
        assert stmt.qubits == ["q0"]

    def test_op_none_is_valid(self):
        stmt = QcdlStatement.model_validate({})
        assert stmt.op is None

    def test_op_valid_identifiers(self):
        for op in ["h", "cx", "_if", "c_if_else", "my_gate2", "_", "__init__"]:
            stmt = QcdlStatement.model_validate({"op": op})
            assert stmt.op == op

    def test_op_invalid_raises(self):
        for bad in ["1h", "h-gate", "cx!", "my gate", "op.name"]:
            with pytest.raises(ValidationError):
                QcdlStatement.model_validate({"op": bad})

    def test_extra_fields_preserved(self):
        """Extra keys (e.g. card_name) round-trip via model_dump()."""
        raw = {"op": "rx", "qubit": "q0", "kwargs": {"theta": 1.5, "card_name": "c0"}}
        stmt = QcdlStatement.model_validate(raw)
        dumped = stmt.model_dump()
        assert dumped["op"] == "rx"
        assert dumped["kwargs"]["card_name"] == "c0"

    def test_sparse_round_trip(self, sparse_statement):
        """exclude_unset=True with exclude_computed_fields=True reproduces the
        sparse add_statement output."""
        model = QcdlStatement.model_validate(sparse_statement)
        dumped = model.model_dump(exclude_unset=True, exclude_computed_fields=True)
        dumped.pop("name", None)  # backward-compat alias — not in source fixture
        assert dumped == sparse_statement

    def test_dense_round_trip(self, minimal_statement):
        """All fields present round-trip without loss."""
        model = QcdlStatement.model_validate(minimal_statement)
        dumped = model.model_dump()
        dumped.pop("name", None)  # backward-compat alias — not in source fixture
        assert dumped == minimal_statement

    def test_call_qubits_populated_from_call_qubits_key(self):
        """caller_qubits is populated from the explicit 'caller_qubits' key."""
        stmt = QcdlStatement.model_validate(
            {"op": "my_proc", "qubit": None, "caller_qubits": ["q0", "q1"]}
        )
        assert stmt.caller_qubits == ["q0", "q1"]

    def test_call_qubits_serializes_as_call_qubits_key(self):
        """caller_qubits appears as 'caller_qubits' in model_dump(); qubits holds
        the computed list."""
        stmt = QcdlStatement.model_validate(
            {"op": "my_proc", "qubit": None, "caller_qubits": ["q0", "q1"]}
        )
        dumped = stmt.model_dump()
        assert "caller_qubits" in dumped
        assert dumped["caller_qubits"] == ["q0", "q1"]
        assert "qubits" in dumped

    def test_qubits_in_serialized_output(self):
        """The computed qubits field is included in model_dump()."""
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        dumped = stmt.model_dump()
        assert "qubits" in dumped
        assert dumped["qubits"] == ["q0"]
        assert "caller_qubits" in dumped

    def test_args_and_qubits_list(self):
        """qubits (computed) aggregates from all sources, deduped."""
        stmt = QcdlStatement.model_validate(
            {"op": "cx", "qubit": "q0", "args": ["q0", "q1"], "qubits": ["q0", "q1"]}
        )
        assert stmt.args == ["q0", "q1"]
        assert stmt.qubits == ["q0", "q1"]

    # --- qubits computed field ----------------------------------------

    def test_qubits_gate_includes_qubit_field(self):
        """qubits includes the qubit field for a simple gate."""
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        assert stmt.qubits == ["q0"]

    def test_qubits_from_arg(self):
        """qubits includes qubit-like strings in args for non-opaque ops."""
        stmt = QcdlStatement.model_validate({"op": "cx", "qubit": "q0", "args": ["q1"]})
        assert "q0" in stmt.qubits
        assert "q1" in stmt.qubits

    def test_qubits_from_call_qubits_field(self):
        """qubits includes entries from the explicit caller_qubits field."""
        stmt = QcdlStatement.model_validate(
            {"op": "my_proc", "qubit": None, "caller_qubits": ["q0", "q1"]}
        )
        assert "q0" in stmt.qubits
        assert "q1" in stmt.qubits

    def test_qubits_from_kwargs_qubits_key(self):
        """qubits includes names from kwargs['qubits']."""
        stmt = QcdlStatement.model_validate(
            {"op": "sync", "qubit": "q0", "kwargs": {"qubits": ["q1", "q2"]}}
        )
        assert "q0" in stmt.qubits
        assert "q1" in stmt.qubits
        assert "q2" in stmt.qubits

    def test_qubits_deduped(self):
        """qubits never contains duplicates even if the same name appears
        in multiple sources."""
        stmt = QcdlStatement.model_validate(
            {"op": "h", "qubit": "q0", "qubits": ["q0"]}
        )
        assert stmt.qubits.count("q0") == 1

    def test_qubits_includes_couplers(self):
        """qubits includes coupler modules (not just qubits)."""
        stmt = QcdlStatement.model_validate({"op": "cz", "qubit": "q0", "args": ["c0"]})
        assert "q0" in stmt.qubits
        assert "c0" in stmt.qubits

    def test_qubits_opaque_for_comment(self):
        """For comment, args are opaque and must not be extracted."""
        stmt = QcdlStatement.model_validate(
            {"op": "comment", "qubit": "q0", "args": ["q1"]}
        )
        assert stmt.qubits == ["q0"]
        assert "q1" not in stmt.qubits

    def test_qubits_opaque_for_if(self):
        """For If, args are opaque."""
        stmt = QcdlStatement.model_validate({"op": "If", "qubit": "q0", "args": ["q1"]})
        assert stmt.qubits == ["q0"]

    def test_qubits_opaque_for_c_if(self):
        """For c_if, args are opaque."""
        stmt = QcdlStatement.model_validate(
            {"op": "c_if", "qubit": "q0", "args": ["q1"]}
        )
        assert stmt.qubits == ["q0"]

    def test_qubits_if_else_kwargs(self):
        """For If/Else/Endif, qubit-valued kwargs (except condition/source_qubit)
        are added to qubits."""
        stmt = QcdlStatement.model_validate(
            {
                "op": "If",
                "qubit": "q0",
                "kwargs": {"condition": 1, "target_qubit": "q1"},
            }
        )
        assert "q0" in stmt.qubits
        assert "q1" in stmt.qubits
        # condition is not a qubit, should not appear
        assert 1 not in stmt.qubits  # type: ignore[operator]

    def test_qubits_procedure_call_no_qubit_field(self):
        """Procedure calls (qubit=None) still populate qubits from caller_qubits."""
        stmt = QcdlStatement.model_validate(
            {"op": "my_proc", "qubit": None, "caller_qubits": ["q0", "q1", "q2"]}
        )
        assert stmt.is_procedure_call
        assert sorted(m.name for m in stmt.qubits) == ["q0", "q1", "q2"]


# ---------------------------------------------------------------------------
# QcdlSignature
# ---------------------------------------------------------------------------


class TestQcdlSignature:
    def test_valid_construction(self, minimal_signature):
        sig = QcdlSignature.model_validate(minimal_signature)
        assert sig.qcdl_operator is None
        assert sig.qubits == ["q0"]
        assert sig.qubits_used == ["q0"]
        assert sig.args == []
        assert sig.kwargs == {}

    def test_null_qcdl_operator_allowed(self, minimal_signature):
        """qcdl_operator may legitimately be None."""
        minimal_signature["qcdl_operator"] = None
        sig = QcdlSignature.model_validate(minimal_signature)
        assert sig.qcdl_operator is None

    def test_named_qcdl_operator(self, minimal_signature):
        minimal_signature["qcdl_operator"] = "h"
        sig = QcdlSignature.model_validate(minimal_signature)
        assert sig.qcdl_operator == "h"

    def test_missing_required_field_raises(self, minimal_signature):
        del minimal_signature["qubits"]
        with pytest.raises(ValidationError):
            QcdlSignature.model_validate(minimal_signature)

    def test_extra_keys_rejected(self, minimal_signature):
        """extra='forbid' mirrors the strict TypedDict (no extra_items)."""
        minimal_signature["unknown_key"] = "bad"
        with pytest.raises(ValidationError):
            QcdlSignature.model_validate(minimal_signature)

    def test_round_trip(self, minimal_signature):
        model = QcdlSignature.model_validate(minimal_signature)
        assert model.model_dump() == minimal_signature


# ---------------------------------------------------------------------------
# QcdlProcedureDef
# ---------------------------------------------------------------------------


class TestQcdlProcedureDef:
    def test_valid_construction(self, minimal_procedure_def):
        proc = QcdlProcedureDef.model_validate(minimal_procedure_def)
        assert proc.statement_hash == "abc123"
        assert len(proc.statements) == 1
        assert isinstance(proc.statements[0], QcdlStatement)
        assert isinstance(proc.signature, QcdlSignature)

    def test_empty_statements_allowed(self, minimal_procedure_def):
        minimal_procedure_def["statements"] = []
        proc = QcdlProcedureDef.model_validate(minimal_procedure_def)
        assert proc.statements == []

    def test_nested_statement_validation(self, minimal_procedure_def):
        """Nested dicts are coerced to QcdlStatement instances."""
        minimal_procedure_def["statements"] = [
            {"op": "h", "qubit": "q0"},
            {"op": "measure", "qubit": "q0"},
        ]
        proc = QcdlProcedureDef.model_validate(minimal_procedure_def)
        assert all(isinstance(s, QcdlStatement) for s in proc.statements)

    def test_bad_nested_signature_propagates_error(self, minimal_procedure_def):
        """ValidationError from a nested model bubbles up."""
        minimal_procedure_def["signature"]["qubits"] = "not-a-list"
        with pytest.raises(ValidationError):
            QcdlProcedureDef.model_validate(minimal_procedure_def)

    def test_missing_statement_hash_is_none(self, minimal_procedure_def):
        del minimal_procedure_def["statement_hash"]
        model = QcdlProcedureDef.model_validate(minimal_procedure_def)
        assert model.statement_hash is None

    def test_extra_keys_rejected(self, minimal_procedure_def):
        minimal_procedure_def["extra_field"] = "nope"
        with pytest.raises(ValidationError):
            QcdlProcedureDef.model_validate(minimal_procedure_def)

    def test_round_trip(self, minimal_procedure_def):
        expected = {
            k: v for k, v in minimal_procedure_def.items() if k != "statement_hash"
        }
        model = QcdlProcedureDef.model_validate(minimal_procedure_def)
        assert _strip_compat_name(model.model_dump()) == expected


# ---------------------------------------------------------------------------
# Qcdl
# ---------------------------------------------------------------------------


class TestQcdl:
    def test_valid_construction(self, minimal_qcdl):
        model = Qcdl.model_validate(minimal_qcdl)
        assert isinstance(model.program, QcdlProcedureDef)
        assert model.procedures == {}
        assert model.next_indices == {}

    def test_named_procedure_accepted(self, minimal_qcdl, minimal_procedure_def):
        minimal_qcdl["procedures"]["my_proc"] = minimal_procedure_def
        model = Qcdl.model_validate(minimal_qcdl)
        assert "my_proc" in model.procedures
        assert isinstance(model.procedures["my_proc"], QcdlProcedureDef)
        assert model.procedures["my_proc"].statement_hash == "abc123"

    def test_extra_fields_preserved(self, minimal_qcdl):
        """Extra top-level keys are preserved via extra="allow"."""
        minimal_qcdl["compiler_hint"] = "fast_path"
        model = Qcdl.model_validate(minimal_qcdl)
        assert model.model_dump()["compiler_hint"] == "fast_path"

    def test_missing_required_field_raises(self, minimal_qcdl):
        del minimal_qcdl["procedures"]
        with pytest.raises(ValidationError):
            Qcdl.model_validate(minimal_qcdl)

    def test_next_indices_round_trip(self, minimal_qcdl):
        minimal_qcdl["next_indices"] = {"label": 3, "axis": 7}
        model = Qcdl.model_validate(minimal_qcdl)
        assert model.next_indices == {"label": 3, "axis": 7}

    def test_round_trip(self, minimal_qcdl):
        expected = copy.deepcopy(minimal_qcdl)
        expected["program"].pop("statement_hash", None)
        model = Qcdl.model_validate(minimal_qcdl)
        assert _strip_compat_name(model.model_dump()) == expected

    def test_validate_from_qcdl_circuit(self):
        """Integration: @qcdl returns a Qcdl directly."""

        @qcdl()
        def my_circuit(q0):
            q0.h()
            q0.measure()

        model = my_circuit()

        assert isinstance(model, Qcdl)
        assert isinstance(model.program, QcdlProcedureDef)
        assert all(isinstance(s, QcdlStatement) for s in model.program.statements)
        ops = [s.op for s in model.program.statements]
        assert "h" in ops
        assert "measure" in ops

    def test_model_dump_compatible_with_compiler(self):
        """model_dump() output has the expected compiler-facing keys."""

        @qcdl()
        def my_circuit(q0):
            q0.h()
            q0.measure()

        model = my_circuit()
        dumped = model.model_dump()

        assert set(dumped.keys()) >= {"program", "procedures", "next_indices"}
        assert "statements" in dumped["program"]
        assert "statement_hash" not in dumped["program"]
        assert "signature" in dumped["program"]

    def test_to_model_dump_contains_no_pydantic_models(self):
        """model_dump() on the Qcdl returned by @qcdl must produce pure
        Python — no BaseModel instances anywhere.

        Exercises both the main program and a named procedure so that all
        serialization paths (statements, signatures, qubits, qubits_used) are
        covered.
        """

        def inner(qa, qb):
            qa.h()
            qa.cnot(qb)

        @qcdl(2)
        def my_circuit(q0, q1):
            p = procedure(inner)
            p(q0, q1)
            q0.measure()

        raw = my_circuit().model_dump()

        def _find_models(obj, path: str = "") -> list[str]:
            if isinstance(obj, BaseModel):
                return [path or "<root>"]
            found: list[str] = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    found.extend(_find_models(v, f"{path}[{k!r}]"))
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    found.extend(_find_models(v, f"{path}[{i}]"))
            return found

        leaks = _find_models(raw)
        assert leaks == [], (
            f"Pydantic models leaked into model_dump() output at: {leaks}"
        )


class TestQcdlStatementStatementFeatures:
    """Derived properties and methods on QcdlStatement."""

    # --- qubit_name ----------------------------------------------------

    def test_qubit_name_alias(self):
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        assert stmt.qubit_name == "q0"

    def test_qubit_name_none_for_procedure_call(self):
        stmt = QcdlStatement.model_validate({"op": "proc", "qubit": None})
        assert stmt.qubit_name is None

    # --- is_procedure_call ---------------------------------------------

    def test_is_procedure_call_true_when_name_none(self):
        stmt = QcdlStatement.model_validate(
            {"op": "my_proc", "qubit": None, "qubits": ["q0"]}
        )
        assert stmt.is_procedure_call is True

    def test_is_procedure_call_false_for_gate(self):
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        assert stmt.is_procedure_call is False

    # --- modules -------------------------------------------------------

    def test_modules_from_name(self):
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        assert stmt.modules == ["q0"]

    def test_modules_from_qubit_arg(self):
        """For most ops, qubit-like args are extracted into modules."""
        stmt = QcdlStatement.model_validate({"op": "cx", "qubit": "q0", "args": ["q1"]})
        assert "q0" in stmt.modules
        assert "q1" in stmt.modules

    def test_modules_excludes_qubit_arg_for_comment(self):
        """For 'comment', args are opaque and must NOT be extracted."""
        stmt = QcdlStatement.model_validate(
            {"op": "comment", "qubit": "q0", "args": ["q1"]}
        )
        assert stmt.modules == ["q0"]
        assert "q1" not in stmt.modules

    def test_modules_excludes_qubit_arg_for_if(self):
        """For 'If', args are opaque."""
        stmt = QcdlStatement.model_validate({"op": "If", "qubit": "q0", "args": ["q1"]})
        assert stmt.modules == ["q0"]

    def test_modules_excludes_qubit_arg_for_c_if(self):
        stmt = QcdlStatement.model_validate(
            {"op": "c_if", "qubit": "q0", "args": ["q1"]}
        )
        assert stmt.modules == ["q0"]

    def test_modules_no_duplicates(self):
        stmt = QcdlStatement.model_validate(
            {"op": "h", "qubit": "q0", "qubits": ["q0"]}
        )
        assert stmt.modules.count("q0") == 1

    def test_modules_includes_coupler(self):
        stmt = QcdlStatement.model_validate({"op": "cz", "qubit": "q0", "args": ["c0"]})
        assert "c0" in stmt.modules

    def test_modules_from_kwargs_qubits_key(self):
        """Qubit names under kwargs['qubits'] are included in modules."""
        stmt = QcdlStatement.model_validate(
            {"op": "sync", "qubit": "q0", "kwargs": {"qubits": ["q1"]}}
        )
        assert "q1" in stmt.modules

    # --- caller_qubits -----------------------------------------------

    def test_caller_qubits_is_empty_for_gate(self):
        """caller_qubits is empty for gate statements; use qubits for all modules."""
        stmt = QcdlStatement.model_validate({"op": "cx", "qubit": "q0", "args": ["q1"]})
        assert stmt.caller_qubits == []
        assert "q0" in stmt.qubits
        assert "q1" in stmt.qubits

    def test_caller_qubits_proc_call_returns_raw_qubits_field(self):
        stmt = QcdlStatement.model_validate(
            {"op": "my_proc", "qubit": None, "caller_qubits": ["q0", "q1"]}
        )
        assert stmt.caller_qubits == ["q0", "q1"]

    # --- qargs / cargs -----------------------------------------------

    def test_qargs_single_qubit(self):
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        assert stmt.qargs == [0]

    def test_qargs_two_qubits(self):
        stmt = QcdlStatement.model_validate({"op": "cx", "qubit": "q0", "args": ["q1"]})
        assert set(stmt.qargs) == {0, 1}

    def test_qargs_excludes_couplers(self):
        stmt = QcdlStatement.model_validate({"op": "cz", "qubit": "q0", "args": ["c0"]})
        assert stmt.qargs == [0]

    def test_cargs_always_none(self):
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        assert stmt.cargs is None

    # --- condition ---------------------------------------------------

    def test_condition_from_first_arg_for_if(self):
        stmt = QcdlStatement.model_validate({"op": "If", "qubit": "q0", "args": [1]})
        assert stmt.condition == 1

    def test_condition_from_kwargs_for_if(self):
        stmt = QcdlStatement.model_validate(
            {"op": "If", "qubit": "q0", "kwargs": {"condition": 1}}
        )
        assert stmt.condition == 1

    def test_condition_last_arg_for_c_if(self):
        stmt = QcdlStatement.model_validate(
            {"op": "c_if", "qubit": "q1", "args": ["x", "q0"]}
        )
        assert stmt.condition == "q0"

    def test_condition_last_arg_for_c_if_else(self):
        stmt = QcdlStatement.model_validate(
            {"op": "c_if_else", "qubit": "q0", "args": [True]}
        )
        assert stmt.condition is True

    def test_condition_raises_for_non_conditional_op(self):
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        with pytest.raises(QCDLInternalError):
            _ = stmt.condition

    def test_condition_raises_for_if_without_condition(self):
        stmt = QcdlStatement.model_validate({"op": "If", "qubit": "q0"})
        with pytest.raises(QCDLInternalError):
            _ = stmt.condition

    # --- simple_desc -------------------------------------------------

    def test_simple_desc_cpu(self):
        stmt = QcdlStatement.model_validate(
            {"op": "cpu", "qubit": "q0", "args": ["x = 1\ny = 2"]}
        )
        assert stmt.simple_desc() == "x = 1; y = 2"

    def test_simple_desc_if(self):
        stmt = QcdlStatement.model_validate({"op": "If", "qubit": "q0", "args": [1]})
        assert stmt.simple_desc() == "If(1)"

    def test_simple_desc_else_no_goto(self):
        stmt = QcdlStatement.model_validate({"op": "Else", "qubit": "q0"})
        assert stmt.simple_desc() == "Else"

    def test_simple_desc_else_with_goto(self):
        stmt = QcdlStatement.model_validate(
            {"op": "Else", "qubit": "q0", "kwargs": {"goto": "loop_end"}}
        )
        assert stmt.simple_desc() == "Else(goto=loop_end)"

    def test_simple_desc_label(self):
        stmt = QcdlStatement.model_validate(
            {"op": "label", "qubit": "q0", "kwargs": {"label": "loop_start"}}
        )
        assert stmt.simple_desc() == "label(loop_start)"

    def test_simple_desc_generic_falls_back_to_str(self):
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        result = stmt.simple_desc()
        assert "h" in result
        assert "q0" in result

    # --- __str__ -----------------------------------------------------

    def test_str_gate_contains_op_and_qubit(self):
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        s = str(stmt)
        assert "h" in s
        assert "q0" in s

    def test_str_gate_with_arg(self):
        stmt = QcdlStatement.model_validate({"op": "rx", "qubit": "q0", "args": [1.5]})
        s = str(stmt)
        assert "rx" in s
        assert "1.5" in s

    def test_str_procedure_call(self):
        stmt = QcdlStatement.model_validate(
            {"op": "my_proc", "qubit": None, "caller_qubits": ["q0", "q1"]}
        )
        s = str(stmt)
        assert "my_proc" in s
        assert "q0" in s
        assert "q1" in s

    # --- reassign_arg / reassign_kwarg -------------------------------

    def test_reassign_arg_mutates_args(self):
        stmt = QcdlStatement.model_validate({"op": "rx", "qubit": "q0", "args": [1.5]})
        stmt.reassign_arg(0, 2.0)
        assert stmt.args[0] == 2.0

    def test_reassign_kwarg_sets_new_value(self):
        stmt = QcdlStatement.model_validate(
            {"op": "p", "qubit": "q0", "kwargs": {"theta": 1.5}}
        )
        stmt.reassign_kwarg("theta", 2.0)
        assert stmt.kwargs["theta"] == 2.0

    def test_reassign_kwarg_adds_new_key(self):
        stmt = QcdlStatement.model_validate({"op": "h", "qubit": "q0"})
        stmt.reassign_kwarg("new_key", "new_val")
        assert stmt.kwargs["new_key"] == "new_val"

    # --- integration: real circuits -----------------------------------

    def test_cx_modules_and_qargs(self):
        """modules and qargs are correct for a real cx statement."""

        @qcdl()
        def circuit(q0, q1):
            q0.cx(q1)

        model = circuit().program.statements[0]

        assert "q0" in model.modules
        assert "q1" in model.modules
        assert model.qargs == [m.index for m in model.modules if m.is_qubit]
        assert not model.is_procedure_call

    def test_comment_args_are_opaque(self):
        """comment args must not be interpreted as qubit names."""

        @qcdl()
        def circuit(q0, q1):
            q0.comment("q1 not a qubit here")
            q0.measure()

        model = circuit().program.statements[0]

        assert model.op == "comment"
        assert "q1" not in model.modules
        assert model.modules == ["q0"]
