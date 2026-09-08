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
Render QCDL programs for terminals and notebooks.

Presentation only; the QCDLv2 text itself is produced by
:mod:`dwave.gate.qcdl.transformer`. Both `black` and `IPython` are optional.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dwave.gate.qcdl.transformer import transform_program_to_qcdl_str, transform_qcdl

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
    from dwave.gate.qcdl.circuit import QCDLV2
    from dwave.gate.qcdl.models import QCDLProgram

__all__ = ["blacken_qcdl_str", "display_qcdl", "print_qcdl"]


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

            from dwave.gate.qcdl import qcdl
            from dwave.gate.utils.display import display_qcdl

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
