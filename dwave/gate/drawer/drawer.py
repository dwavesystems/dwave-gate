# Copyright 2023 D-Wave Systems Inc.
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

import inspect
from shutil import get_terminal_size
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from dwave.gate.drawer.characters import *
from dwave.gate.drawer.column import BarrierColumn, CircuitColumn

if TYPE_CHECKING:
    from dwave.gate import Circuit
    from dwave.gate.operations import Operation


__all__ = [
    "CircuitDrawer",
]


class CircuitDrawer:
    """Class for drawing a circuit in the terminal.

    Contains one or more columns, each representing a set of gates which can be applied
    simultaneously to different qubits. This class provides methods for printing/drawing circuits in
    the terminal and for converting a :class:`dwave.gate.circuit.Circuit` object into columns.

    Args:
        num_qubits: The number of qubits in the circuit
    """

    def __init__(self, num_qubits: int) -> None:
        self._columns = []
        self._num_qubits = num_qubits

    @property
    def num_qubits(self):
        """Number of qubits drawn."""
        return self._num_qubits

    def add_column_data(
        self,
        target_names: Union[str, Sequence[str]],
        targets: Sequence[int],
        controls: Optional[Sequence[int]] = None,
        build_kwargs: Dict[str, Any] = None,
    ) -> None:
        """Helper method to add a column using data necessary to construct it.

        Instead of having to create a :class:`CircuitColumn` and add it
        manually, this method does that automatically using only the necessary column data.

        Args:
            target_names: Names for the target boxes. Some names are automatically converted to
                relevant symbols (see ``column.single_char_ops`` for details).
            targets: The qubits on which the target boxes should be applied.
            controls: The control qubits.
            build_kwargs: Any arguments that should be passed to the
                :meth:`CircuitColumn.build` method.
        """
        build_kwargs = build_kwargs or {}

        col = CircuitColumn(target_names=target_names, targets=targets, controls=controls)
        col.build(**build_kwargs)

        self.add_column(col)

    def add_column(self, column: CircuitColumn) -> None:
        """Add a column to the drawer.

        Args:
            column: The ``CircuitColumn`` to add.
        """
        if column.num_qubits > self.num_qubits:
            raise ValueError(
                f"Circuit drawer only has {self.num_qubits} qubits. Cannot add "
                f"operations on qubit {column.num_qubits}."
            )
        self._columns.append(column)

    def add_operation(
        self,
        op: Operation,
        qubits: Optional[Sequence[int]] = None,
        circuit: Optional[Circuit] = None,
    ) -> None:
        """Add an operation to the drawer.

        Uses :meth:`CircuitColumn.from_operation` to create a column with a single operation.
        By passing a list of integers to ``qubits`` the operation qubits are overridden which
        in turn makes any ``circuit`` passed to be ignored. Otherwise, the operation qubits are
        mapped to their respective positions in the ``circuit``.

        Args:
            op: The operation to add.
            qubits: Optional qubits on which the operation is applied. If not passed,
                the qubits are retrieved from the operation. Overides the operation qubits
                ignoring any circuit passed to the method.
            circuit: Optional circuit containing relevant qubit registers. If ``None`` and
                no qubits are passed, then consecutive qubits are used starting at 0. Must
                contain the same qubits as the operation.
        """
        connects = None
        if circuit and not qubits:
            max_qubit = max(circuit.find_qubit(qb)[1] for qb in op.qubits)
            if max_qubit >= self.num_qubits:
                raise ValueError(
                    f"Circuit drawer only has {self.num_qubits} qubits. Cannot add "
                    f"operations on qubit {max_qubit}."
                )

            if hasattr(op, "control"):
                connects = [
                    (circuit.find_qubit(c)[1], circuit.find_qubit(t)[1])
                    for c in op.control
                    for t in op.target
                ]

        elif hasattr(op, "control"):
            control = qubits[: op.num_control] if qubits else range(op.control)
            target = qubits[op.num_control :] if qubits else range(op.control)
            connects = [(c, t) for c in control for t in target]

        col = CircuitColumn.from_operation(op, qubits, circuit)

        if op.name == "Barrier":
            col.build()
        else:
            col.build(connects=connects)

        self.add_column(col)

    def _as_str(self):
        """Returns the circuit drawer object as a printable string."""
        nint = len(str(self.num_qubits))
        # construct all qubit wires starting with 'q0', 'q1', ..., 'q42', etc.
        lines = [spc * (2 + nint)] + [
            element
            for sub in zip(
                [f"q{i}{spc * (1 + nint - len(str(i)))}" for i in range(self.num_qubits)],
                [spc * (2 + nint)] * self.num_qubits,
            )
            for element in sub
        ]
        """Alternative qubit representations (commented out below)"""
        # # Lines start with 'q0' (as in the 0-initialized state)
        # lines = [spc * 3] + [f"q0{spc}", spc * 3] * self.num_qubits
        # # Lines start with '|0>' (doesn't look good in Jupyter)
        # lines = [spc * 4] + [f"{vbar}0{rang} ", spc * 4] * self.num_qubits

        for c in self._columns:
            start_idx = min(c.targets + c.controls)

            for i, l in enumerate(c.lines):
                lines[start_idx * 2 + i] += l

            # fill all other qubits
            lines = self._fill_lines(lines, self.num_qubits, start_idx)

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Creates the representation used for Jupyter notebook outputs."""
        return (
            '<pre style="word-wrap: normal;'
            "line-height: 1.2;"
            'font-family: &quot;Courier New">'
            "%s</pre>" % self._as_str()
        )

    def __repr__(self) -> str:
        """The circuit drawer representation."""
        return self._as_str()

    def __str__(self) -> str:
        """The circuit drawer representation."""
        try:
            from IPython.core.getipython import get_ipython
            from IPython.display import display
        except ImportError:
            interactive_shell = False
        else:
            # inspect the previous entry in the stack to check for 'print' call
            called_with_print = inspect.stack()[1][4][0].startswith("print(")
            # check whether in Jupyter notebook and switch to HTML 'display'
            interactive_shell = "InteractiveShell" in get_ipython().__class__.__name__
            if called_with_print and interactive_shell:
                display(self)
                return ""

        return self._as_str()

    @staticmethod
    def draw_circuit(circuit: Circuit) -> None:
        """Draws a :class:`Circuit`.

        Adds all operations in the circuit one-by-one.

        Args:
            circuit: Circuit to be drawn.
        """
        drawer = CircuitDrawer(circuit.num_qubits)

        for op in circuit.circuit:
            drawer.add_operation(op, circuit=circuit)

        return drawer

    @staticmethod
    def _fill_lines(lines, num_qubits, last_idx):
        """Fill empty lines so that the circuit drawing is justified
        after each column application.

        Args:
            lines: The lines which have had columns added to.
            num_qubits: The number of qubits in total.
            last_idx: TODO
        """
        longest_line = len(lines[last_idx * 2])

        for j in range(num_qubits):
            lines[2 * j + 1] = lines[2 * j + 1].ljust(longest_line, hbar)
            lines[2 * j] = lines[2 * j].ljust(longest_line)

        lines[-1] = lines[-1].ljust(longest_line)

        return lines
