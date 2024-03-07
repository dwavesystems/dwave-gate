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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Self, Sequence, Tuple, Union

from dwave.gate.drawer.characters import *

if TYPE_CHECKING:
    from dwave.gate.circuit import Circuit
    from dwave.gate.operations import Operation


__all__ = [
    "CircuitColumn",
]


single_char_ops = {
    "SWAP": swp,
}


class Column(ABC):
    """Abstract base class for handling and drawing columns.

    Args:
        targets: The target qubits on which operations are drawn.
        controls: Optional control qubits.
    """

    def __init__(self, targets: Sequence[int], controls: Optional[Sequence[int]] = None) -> None:
        self._targets = targets
        self._controls = controls or []

        self._lines = []

    @property
    def lines(self) -> Sequence[str]:
        """The lines of the column."""
        return self._lines

    @property
    def targets(self) -> Sequence[int]:
        """The target qubit indices."""
        return self._targets

    @property
    def controls(self) -> Sequence[int]:
        """The control qubit indices."""
        return self._controls

    @property
    def num_qubits(self) -> int:
        """The number of qubits."""
        return max(self.targets) + 1

    @property
    def min_qubit(self) -> int:
        """The minimum qubit index with an operation (target or control)."""
        return min(self.targets)

    @property
    def max_qubit(self) -> int:
        """The maximum qubit index with an operation (target or control)"""
        return max(self.targets)

    def _as_str(self) -> str:
        """Returns the circuit drawer object as a printable string."""
        return "\n".join(self.lines)

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

    @abstractmethod
    def build(self) -> Column:
        """Abstract method for contstucting the column string from data."""


class CircuitColumn(Column):
    """A column representing a set of boxed operations.

    Args:
        target_names: The names of the target operations.
        targets: The target qubit indices.
        control: Optional control qubit indices.
    """

    def __init__(
        self,
        target_names: Union[str, Sequence[str]],
        targets: Sequence[int],
        controls: Optional[Sequence[int]] = None,
    ) -> None:

        if isinstance(target_names, str):
            self._target_names = [target_names] * len(targets)
        else:
            if len(target_names) != len(targets):
                raise ValueError(
                    f"Each target must have a name. Got {len(target_names)} names "
                    f"but have {len(targets)} targets."
                )
            self._target_names = target_names

        """Drawing constants (incl. common string combinations and lines)."""
        self._box_len = self._set_box_len()
        self._center = self._box_len // 2

        self._box_space = self._box_len * spc
        self._box_vbar = (
            self._box_space[: self._center] + vbar + self._box_space[self._center + 1 :]
        )

        self._box_hbar = self._box_len * hbar

        """Control qubits."""
        self._box_tbk = self._box_hbar[: self._center] + tbk + self._box_hbar[self._center + 1 :]
        self._box_bbk = self._box_hbar[: self._center] + bbk + self._box_hbar[self._center + 1 :]
        self._box_blk = self._box_hbar[: self._center] + blk + self._box_hbar[self._center + 1 :]

        """Connecting qubits (mainly for controlled ops and swaps)."""
        self._box_crs = self._box_hbar[: self._center] + crs + self._box_hbar[self._center + 1 :]

        super().__init__(targets, controls)

    @property
    def target_names(self) -> Sequence[str]:
        """Target names used to represent drawn gates."""
        return self._target_names

    @property
    def num_qubits(self) -> int:
        """The number of drawn qubits."""
        return max(self.targets + self.controls) + 1

    @property
    def min_qubit(self) -> int:
        """The minimum qubit index in the drawing."""
        return min(self.targets + self.controls)

    @property
    def max_qubit(self) -> int:
        """The maximum circuit index in the drawing."""
        return max(self.targets + self.controls)

    def _set_box_len(self):
        """Helper function to get the box length."""
        box_len = 3
        for name in self._target_names:
            l = len(name) + 4
            if name not in single_char_ops and l > box_len:
                box_len = l

        return box_len

    def get_connects(self, current: int, connects: Sequence[Tuple[int, int]]) -> Tuple[bool, bool]:
        """Two booleans representing whether the current qubit is connect up and/or down.

        Args:
            current: The qubit index which is to be checked for connecting neighbours.
            connects: Sequence of connected qubits.

        Returns:
            Tuple[bool, bool]: Two booleans for an up and down connection respectively.
        """
        if not connects:
            return False, False

        _connect_up = _connect_down = False
        for c in connects:
            if current in c:
                if self._last(current) in c:
                    _connect_up = True

                if self._next(current) in c:
                    _connect_down = True

        target_above = current in self._targets and current - 1 in self._targets
        target_below = current in self._targets and current + 1 in self._targets

        connect_up = current != self.min_qubit and (_connect_up and not target_above)
        connect_down = current != self.max_qubit and (_connect_down and not target_below)

        return connect_up, connect_down

    def _assert_connects(self, connects) -> None:
        """Assert that the connects are valid."""
        if not connects:
            return

        active_qubits = self.controls + self.targets

        _connects_ranges = map(lambda tup: tuple(range(*sorted(tup))[1:]), connects)

        blocked_qubits = set(active_qubits).intersection(i for c in _connects_ranges for i in c)
        non_active_qubits_used = set(qb for c in connects for qb in c).difference(active_qubits)

        if non_active_qubits_used or blocked_qubits:
            raise ValueError(
                "Connects can only be made between active qubits "
                "(i.e., qubits that are either controls or targets)."
            )

    def build(self, connects: Optional[Sequence[Tuple[int, int]]] = None) -> CircuitColumn:
        """Build the circuit column populating the ``line`` attribute.

        Compiles the ``CircuitColumn`` in place and also returns itself.

        Args:
            connects: Sequence of consecutive number pairs corresponding to connected qubits.

        Returns:
            CircuitColumn: The compiled ``CircuitColumn`` containing the correct lines for printing.
        """
        self._assert_connects(connects)

        # replace already built lines
        if self._lines:
            self._lines = []

        t = 0
        for i in range(self.min_qubit, self.max_qubit + 1):

            connect_up, connect_down = self.get_connects(i, connects)
            if i in self._controls:
                self._add_control(i, connect_up, connect_down)

            elif i in self._targets:
                self._add_target(i, t, connect_up, connect_down)

                # increment target idx by one
                t += 1

            else:
                is_empty_line = i - 1 not in self._targets and i - 1 not in self._controls
                if connects and any(i in range(*sorted(c)) for c in connects):
                    if is_empty_line:
                        self._lines.append(self._box_vbar)
                    self._lines.append(self._box_crs)
                else:
                    if is_empty_line:
                        self._lines.append(self._box_space)
                    self._lines.append(self._box_hbar)

        return self

    def _last(self, current: int) -> int:
        """Get the index of the previous target or control qubit.

        Args:
            current: The index of the current qubit.

        Returns:
            The index of the previous target or control qubit.
        """
        for i in range(current - 1, -1, -1):
            if i in self.targets or i in self.controls:
                return i

    def _next(self, current: int) -> int:
        """Get the index of the next target or control qubit.

        Args:
            current: The index of the current qubit.

        Returns:
            The index of the next target or control qubit.
        """
        for i in range(current + 1, self.num_qubits + 1):
            if i in self.targets or i in self.controls:
                return i

    def _add_control(self, current: int, connect_up: bool, connect_down: bool) -> None:
        """Add a control box to the column lines.

        Args:
            current: The index of the current qubit.
            connect_up: Whether the box has a connection at the top.
            connect_down: Whether the box has a connection at the bottom.
        """
        if current - 1 in self._controls:
            del self._lines[-1]

        if connect_up and connect_down:
            self._lines.extend((self._box_vbar, self._box_blk, self._box_vbar))
        elif connect_down:
            self._lines.extend((self._box_space, self._box_tbk, self._box_vbar))
        elif connect_up:
            self._lines.extend((self._box_vbar, self._box_bbk, self._box_space))
        else:
            # self._lines.extend((self._box_space, self._box_blk, self._box_space))
            raise ValueError(f"Unconnected control on qubit {current}.")

        # if last qubit was a target, cut away top of control qubit
        if current - 1 in self._targets:
            del self._lines[-3]

    def _add_target(self, current: int, t: int, connect_up: bool, connect_down: bool) -> None:
        """Add a target box to the column lines.

        Args:
            current: The index of the current qubit.
            t: The number of the target qubit.
            connect_up: Whether the box has a connection at the top.
            connect_down: Whether the box has a connection at the bottom.
        """

        if current - 1 in self._controls:
            del self._lines[-1]

        target_box = self.box(
            self._target_names[t], connect_up=connect_up, connect_down=connect_down
        )
        if current - 1 in self._targets:
            # if current box is single-char
            if self._target_names[t] in single_char_ops:
                del target_box[0]

            # if upper box is single-char
            elif self._target_names[t - 1] in single_char_ops:
                del self._lines[-1]

            # if upper box is smaller than lower box
            elif len(self._target_names[t]) > len(self._target_names[t - 1]):
                del self._lines[-1]
                target_box[0] = self._replace_corners(target_box[0], t, t - 1)

            # if upper box is bigger than lower box
            elif len(self._target_names[t]) < len(self._target_names[t - 1]):
                self._lines[-1] = self._replace_corners(self._lines[-1], t - 1, t)
                del target_box[0]

            # if upper box is the same size as the lower box
            else:
                for a, b in [(ble, rba), (bri, lba), (tco, hbar), (bco, hbar)]:
                    self._lines[-1] = self._lines[-1].replace(a, b)
                del target_box[0]

        self._lines.extend(target_box)

    def _replace_corners(self, line: str, t0: int, t1: int) -> str:
        """Replace box corners with the correct symbol when two targets are touching.

        Args:
            line: The line on which to replace corners.
            t0: The index of the top target.
            t1: The index of the bottom target.

        Returns:
            str: The line with replaced/correct corners.
        """
        lpad_0, rpad_0 = self._get_padding(self._target_names[t0])
        lpad_1, rpad_1 = self._get_padding(self._target_names[t1])

        if lpad_0 == lpad_1:
            lcorner = rba
        elif t0 < t1:
            lcorner = bco
        else:
            lcorner = tco

        if rpad_0 == rpad_1:
            rcorner = lba
        elif t0 < t1:
            rcorner = bco
        else:
            rcorner = tco

        line = line[:lpad_1] + lcorner + line[lpad_1 + 1 :]
        line = line[: -rpad_1 - 1] + rcorner + line[-rpad_1:]
        return line

    def _get_padding(self, name: str) -> Tuple[int, int]:
        """Get the padding around a box with arbitrary labels.

        Args:
            name: The label that is printed in the box.

        Returns:
            Tuple[int, int]: Tuple containing the padding to the left and right of the box.
        """
        # if name is a non-boxed char, set it to -2 (removing the two border tiles)
        # unless length of box is even; then set it to -1 to round up // 2
        if name in single_char_ops:
            if self._box_len % 2 == 0:
                name_len = -2
            else:
                name_len = -1
        else:
            name_len = len(name)

        lpad = (self._box_len - (name_len + 4)) // 2
        rpad = self._box_len - (name_len + 4) - lpad

        return lpad, rpad

    def box(self, name: str, connect_up: bool = False, connect_down: bool = False) -> Sequence[str]:
        """Create a target box with a label.

        Args:
            name: The label for the operation.
            connect_up: Whether the box has a connection at the top.
            connect_down: Whether the box has a connection at the bottom.

        Returns:
            Sequence[str]: The lines which prints the box.
        """
        lpad, rpad = self._get_padding(name)

        box_top = box_bot = hbar * (len(name) + 2)

        lines = []
        if name in single_char_ops:
            connect_top = connect_bot = vbar

            top_line = bot_line = spc * self._box_len
            mid_line = hbar * lpad + f"{hbar}{single_char_ops[name]}{hbar}" + hbar * rpad
            bot_line = spc * lpad + f"{spc}{vbar if connect_down else spc}{spc}" + spc * rpad
        else:
            connect_top = tco
            connect_bot = bco

            top_line = spc * lpad + f"{tle}{box_top}{tri}" + spc * rpad
            mid_line = hbar * lpad + f"{lba} {name} {rba}" + hbar * rpad
            bot_line = spc * lpad + f"{ble}{box_bot}{bri}" + spc * rpad

        if connect_up:
            top_line = top_line[: self._center] + connect_top + top_line[self._center + 1 :]
        if connect_down:
            bot_line = bot_line[: self._center] + connect_bot + bot_line[self._center + 1 :]

        lines.extend((top_line, mid_line, bot_line))
        return lines

    @staticmethod
    def from_operation(
        op: Operation, qubits: Optional[Sequence[int]] = None, circuit: Optional[Circuit] = None
    ) -> Self:
        """Static method for creating a ``CircuitColumn`` from an :meth:`dwave.gate.operation.Operation`.

        Note that only a single operation is currently created per column.

        Args:
            op: The operation from which to create a ``CircuitColumn``.
            qubits: Optional qubits on which the operation should be applied. Overrides any circuit and
                its qubits that are passed to the method.
            circuit: Optional :class:`dwave.gate.Circuit` containing the qubit register on which
                operations may be applied.

        Returns:
            CircuitColumn: ``CircuitColumn`` representation of the operation.
        """

        name = op.name
        controls_ind = []

        targets_ind = list(range(len(qubits or op.qubits)))

        if hasattr(op, "control"):
            controls_ind = qubits[: op.num_control] if qubits else list(range(op.num_control))
            targets_ind = (
                qubits[op.num_control :, op.num_target + 1]
                if qubits
                else list(range(op.num_control, op.num_target + 1))
            )
            name = op.target_operation.name

        elif op.name == "CCX":
            controls_ind = qubits[:2] if qubits else [0, 1]
            targets_ind = qubits[2] if qubits else [2]
            name = "X"

        elif op.name == "SWAP":
            targets_ind = qubits[:2] if qubits else [0, 1]

        elif op.name == "CSWAP":
            controls_ind = qubits[0] if qubits else [0]
            targets_ind = qubits[1:3] if qubits else [1, 2]
            name = "SWAP"

        if circuit is None or qubits:
            return CircuitColumn(name, targets_ind, controls_ind)

        targets = [circuit.find_qubit(op.qubits[qb])[1] for qb in targets_ind]
        controls = [circuit.find_qubit(op.qubits[qb])[1] for qb in controls_ind]

        if name == "Barrier":
            return BarrierColumn(targets)

        return CircuitColumn(name, targets, controls)


class BarrierColumn(Column):
    """A column representing a barrier on one or more qubits.

    Args:
        targets: The target qubit indices.
    """

    def __init__(self, targets: Sequence[int]) -> None:
        self._targets = []
        super().__init__(targets)

    def build(self) -> BarrierColumn:
        """Build the barrier column by populating the ``line`` attribute.

        Compiles the ``BarrierColumn`` in place and also returns itself.

        Returns:
            BarrierColumn: The compiled ``BarrierColumn`` containing the correct lines for printing.
        """

        # replace already built lines
        if self._lines:
            self._lines = []

        t = -1
        for i in range(self.min_qubit, self.max_qubit + 1):

            if i in self._targets:
                top_line = spc * 2 + f"{vbar}" + spc * 2
                mid_line = hbar * 2 + f"{vbar}" + hbar * 2
                bot_line = spc * 2 + f"{vbar}" + spc * 2

                if t == -1:
                    self._lines.extend((top_line, mid_line, bot_line))
                elif i > t + 1:
                    self._lines.extend((top_line, mid_line, bot_line))
                elif i == t + 1:
                    self._lines.extend((mid_line, bot_line))

                t = i

            else:
                is_empty_line = i - 1 not in self._targets and i - 1 not in self._controls
                # TODO: update lines
                if is_empty_line:
                    self._lines.append(" " * 5)
                self._lines.append(hbar * 5)

        return self
