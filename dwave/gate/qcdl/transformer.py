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
transform a JSON-specified circuit from QCDL into qcdl_objects
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .base import VariableExpression
from .qcdl_models import QCDLProgram, QCDLProcedureDef, QCDLStatement
from .qcdl_objects import (
    AsmSync,
    ParserAsmInstruction,
    ParserAsmProgram,
    ProcedureCallStatement,
    Variable,
    format_signature,
    get_indent_change,
)
from .utils import map_container

try:
    import black
except ImportError:
    pass

try:
    from IPython.display import Code, display

    HAVE_IPYTHON = True

except ImportError:
    HAVE_IPYTHON = False

if TYPE_CHECKING:
    from .qcdl_circuit import QCDLV2


def transform_statement(
    statement: QCDLStatement | dict,
    label_suffix: str | None = None,
) -> ParserAsmInstruction | ProcedureCallStatement | AsmSync:
    stmt: QCDLStatement = (
        QCDLStatement.model_validate(statement)
        if isinstance(statement, dict)
        else statement
    )

    qubit_name = stmt.qubit

    # This mocks up how the parse tree object gets the
    # name of the qubit
    def _make_dummy(value: Any) -> Any:
        def obj() -> None:
            return None  # Dummy function

        _obj: Any = obj
        _obj.value = value
        return _obj

    op = stmt.op
    # Copy so that map_container does not mutate the model's fields.
    args = list(stmt.args)
    kwargs = dict(stmt.kwargs)

    if qubit_name is None:
        # this is the case for a procedure being invoked as a function
        # instead of as a Module method
        qubits = [q.name for q in stmt.qubits]

        return ProcedureCallStatement([_make_dummy(op), qubits, args, kwargs])

    qubit_name_str = qubit_name.name

    # handle idiosyncratic statements
    if op == "sync" and args:
        # this is for multi-qubit syncs since python doesn't
        # support the `- q0 q1 -` syntax.
        return AsmSync(set([qubit_name_str] + args), nargs=kwargs)

    card_name = kwargs.pop("card_name", None)

    if label_suffix:
        for k, v in kwargs.items():
            if k in ["label", "goto", "true_goto", "false_goto"]:
                kwargs[k] = v + label_suffix

    # handle other types of objects that may be found as arg and kwarg
    # values. They're all expected to be dicts with a "type" key/value.
    def mapper(value: Any) -> Any:
        if not isinstance(value, dict) or "type" not in value:
            return value
        obj_type = value["type"]
        if obj_type == "variable":
            return Variable(value["variable"])
        elif obj_type == "variable_expression":
            return VariableExpression.deserialize(value)
        return value

    map_container([args, kwargs], map_value=mapper)

    # this is the data structure used to represent an individual instruction
    # the data here will be passed along to a library method or an asm_statement
    # by the compiler (depending on how op is defined)
    items = [_make_dummy(qubit_name_str), dict(op=op, args=args, named_args=kwargs)]
    return ParserAsmInstruction(items, card_name=card_name)


def transform_statements(
    statements: Sequence[QCDLStatement | dict],
    label_suffix: str | None = None,
) -> ParserAsmProgram:
    """Generally this is called for procedures or macros

    label_suffix is a way to give all the statements provided here a common
    scope which is useful if the same label-containing macro is used more than
    once in a procedure.
    """
    return ParserAsmProgram(
        items=[transform_statement(s, label_suffix=label_suffix) for s in statements]
    )


def transform_procedures(
    procedures: Mapping[str, QCDLProcedureDef | dict],
) -> list[dict]:
    procs = []
    for proc_name, proc in procedures.items():
        if isinstance(proc, dict):
            proc = QCDLProcedureDef.model_validate(proc)
        prog = transform_statements(proc.statements)
        procs.append(
            dict(
                procedure=[
                    dict(
                        proc_spec=dict(
                            function_name=proc_name,
                            qubits=[Variable(q.name) for q in proc.signature.qubits],
                            qubits_used=[q.name for q in proc.signature.qubits_used],
                            qcdl_operator=proc.signature.qcdl_operator,
                            arguments=proc.signature.args,
                            named_args=proc.signature.kwargs,
                        )
                    ),
                    prog,
                ]
            )
        )
    return procs


def transform_qcdl(qcdl: QCDLProgram | Mapping[str, Any]) -> dict:
    """This converts the output from QCDL into an object that
    mirrors what the lark parser produces from a .qcdl file.
    """
    if isinstance(qcdl, dict):
        qcdl_model = QCDLProgram.model_validate(qcdl)
    elif isinstance(qcdl, QCDLProgram):
        qcdl_model = qcdl
    else:
        raise TypeError(f"QCDL must be a dictionary, not a {type(qcdl)}")

    program: list[Any] = []

    # this is the "main" block
    program.append(transform_statements(qcdl_model.program.statements))

    # add the procedures (args/kwargs are not passed along, see comment above)
    program.extend(transform_procedures(qcdl_model.procedures))

    ret = dict(quantum_program=program)
    return ret


def transform_statements_to_qcdl_str(
    statements: list[dict], initial_indent: int = 0, indentation: int = 3
) -> str:
    txt = ""
    if isinstance(initial_indent, int):
        indent = initial_indent
    else:
        indent = len(initial_indent)

    for raw_stmt in statements:
        in_proc = False
        if not hasattr(raw_stmt, "_qcdl"):
            in_proc = True
            spec = raw_stmt["procedure"][0]["proc_spec"]
            txt += (
                "\n"
                + " " * indent
                + "begin procedure "
                + format_signature(
                    op=spec["function_name"],
                    qubits=spec["qubits"],
                    args=spec["arguments"],
                    nargs=spec["named_args"],
                )
            )
            stmt = raw_stmt["procedure"][1]
            indent += indentation
        else:
            stmt = raw_stmt

        indent_change = get_indent_change(stmt)
        if indent_change is None:
            txt += stmt._qcdl(indent=" " * indent)

            if in_proc:
                indent -= indentation
                txt += " " * indent + "end procedure\n"
        else:
            indent += indent_change
            # don't display the qcdl_indent instructions

    return txt


def transform_program_to_qcdl_str(qcdl: dict) -> QCDLV2:
    txt = "begin quantum\n"
    txt += transform_statements_to_qcdl_str(qcdl["quantum_program"], initial_indent=3)
    txt += "end quantum\n"
    return txt


def blacken_qcdl_str(qcdl_str: QCDLV2) -> QCDLV2:
    """Treat the code like python and try to use black to reformat it"""
    blackened = []
    indent = 0
    for raw_line in qcdl_str.splitlines():
        if not raw_line.strip():
            blackened.append("\n")
            continue

        if "begin" in raw_line:
            indent += 1
        elif "end" in raw_line:
            indent -= 1
        try:
            line = black.format_str(raw_line, mode=black.Mode())
            line = textwrap.indent(line, prefix="   " * indent)
        except black.InvalidInput:
            line = raw_line + "\n"
        blackened.append(line)
    return "".join(blackened)


def print_qcdl(
    qcdl: QCDLProgram | Mapping[str, Any],
    to_Display: bool = True,
    blacken: bool = False,
    filename: str | None = None,
) -> QCDLV2 | None:
    """Print a QCDl program.

    Args:
        qcdl: A QCDL model or mapping. Typically created by instantiating a
            Python function containing QCDL instructions and annotated with the
            :func:`~dwave.gate.qcdl.qcdl` decorator.
        to_Display: If True, outputs the string to an
            `IPython <https://ipython.org/>`_ terminal. Outside of a
            `Jupyter <https://jupyter.org/>`_ notebook, equivalent to a print
            statement. Set to False to return the string.
        blacken: Apply the `Black <https://pypi.org/project/black/>`_ Python
            formatter to the input QCDL.
        filename: File name to write the string to.

    Returns:
        If displaying, the return is None; otherwise, the "qcdlv2" string.

    Examples:
        See the examples in the :func:`.display_qcdl` function.
    """
    if not HAVE_IPYTHON:
        to_Display = False

    ret = transform_qcdl(qcdl)
    qcdl_str = transform_program_to_qcdl_str(ret)

    if blacken:
        qcdl_str = blacken_qcdl_str(qcdl_str)

    if filename:
        with open(filename, "w") as f:
            f.write(qcdl_str)

    if to_Display:
        # NOTE: this is equivalent to a print statement if we're not in a
        # jupyter context
        display(Code(qcdl_str, language="python"))
        return None
    else:
        return qcdl_str


def display_qcdl(qcdl: QCDLProgram | Mapping[str, Any], **kwargs: Any) -> None:
    """Display formatted QCDL in a `Jupyter <https://jupyter.org/>`_ notebook or
    similar.

    Creates an `IPython Code
    <https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.Code>`_
    object.

    Args:
        qcdl: A QCDL model or mapping. Typically created by instantiating a
            Python function containing QCDL instructions and annotated with the
            :func:`~dwave.gate.qcdl.qcdl` decorator.

    Examples:

        .. testcode::
            :skipif: True       # Not tested because not in JN

            from dwave.gate.qcdl import display_qcdl, qcdl

            @qcdl(1)
            def display_program(q0):

                q0.h()
                q0.measure()

            qcdl_program = display_program()
            display_qcdl(qcdl_program)


        The code above displays the following QCDL program.

        .. testoutput::
            :skipif: True       # Not tested because not in JN
            :options: +NORMALIZE_WHITESPACE

            begin quantum
                q0.h()
                q0.measure()
            end quantum
    """
    qcdl_str = print_qcdl(qcdl, to_Display=False, **kwargs)
    display(Code(qcdl_str, language="python"))
    return
