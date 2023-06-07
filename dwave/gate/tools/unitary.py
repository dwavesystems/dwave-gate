# Copyright 2022 D-Wave Systems Inc.
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

__all__ = [
    "build_unitary",
    "build_controlled_unitary",
]

import itertools
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from dwave.gate.operations.base import ControlledOperation, Operation


#####################
# Circuit functions #
#####################


# TODO: exchange for something better; only here for testing matrix creation
# for custom operations; controlled operations only works with single
# control and target; no support for any other multi-qubit gates
@lru_cache(maxsize=128)
def build_unitary(circuit) -> NDArray:
    """Builds the circuit unitary by multiplying together the operation matrices.

    Args:
        circuit: The circuit for which the unitary should be built.

    Returns:
        NDArray: Unitary matrix representation of the circuit.
    """
    state = np.eye(2**circuit.num_qubits)
    # apply operations from first to last (the order in which they're
    # applied within the context, stored sequentially in the circuit)
    for op in circuit.circuit:
        # check if controlled operation; cannot check isinstance
        # 'Controlled' due to circular import issue
        if hasattr(op, "control"):
            state = _apply_controlled_gate(state, op, circuit.qubits)
        else:
            state = _apply_single_qubit_gate(state, op, circuit.qubits)
    return state


def _apply_single_qubit_gate(state: NDArray, op: Operation, qubits) -> NDArray:
    """Apply a single qubit operation to the state.

    Args:
        state: State on which to apply the operation.
        op: Single-qubit operation to apply (must have ``op.qubits``).
        qubits: List of qubits in the circuit.

    Returns:
        NDArray:
    """
    if op.qubits and op.qubits[0] == qubits[0]:
        mat = op.matrix
    else:
        mat = np.eye(2**op.num_qubits)

    for qb in qubits[1:]:
        if op.qubits and qb == op.qubits[0]:
            mat = np.kron(mat, op.matrix)
        else:
            mat = np.kron(mat, np.eye(2**op.num_qubits))

    return mat @ state


def _apply_controlled_gate(state: NDArray, op: ControlledOperation, qubits) -> NDArray:
    """Apply a controlled qubit gate to the state.

    Args:
        state: State on which to apply the operation.
        op: Controlled operation to apply (must have ``op.control``,
            and ``op.target`` defined).
        qubits: List of qubits in the circuit.

    Returns:
        NDArray: Resulting state vector after application.
    """
    control_idx = [qubits.index(c) for c in op.control or []]
    target_idx = [qubits.index(t) for t in op.target or []]
    controlled_unitary = build_controlled_unitary(
        control_idx, target_idx, op.target_operation.matrix, len(qubits)
    )
    return controlled_unitary @ state


#########################
# Stand-alone functions #
#########################


def build_controlled_unitary(
    control: Union[int, Sequence[int]],
    target: Union[int, Sequence[int]],
    unitary: NDArray,
    num_qubits: Optional[int] = None,
    dtype: DTypeLike = complex,
) -> NDArray:
    """Build the unitary matrix for a controlled operation.

    Args:
        control: Index of control qubit(s).
        target: Index of target qubit. Only a single target supported.
        num_qubits: Total number of qubits.

    Returns:
        NDArray: Unitary matrix representing the controlled operation.
    """
    if isinstance(control, int):
        control = [control]
    if isinstance(target, int):
        target = [target]

    if not set(control).isdisjoint(target):
        raise ValueError("Control qubits and target qubit cannot be the same.")

    max_ = max(itertools.chain.from_iterable((control, target)))
    if isinstance(num_qubits, int) and num_qubits <= max_:
        raise ValueError(
            f"Total number of qubits {num_qubits} must be larger or equal "
            f"to the largest qubit index {max_}."
        )

    # TODO: add support for multiple targets
    if len(target) != 1:
        raise NotImplementedError("Multiple target not currently supported.")

    # if None, set number of qubits to the max control/target value + 1
    num_qubits = num_qubits or max_ + 1

    state = np.eye(2**num_qubits, dtype=dtype)

    # create all control and target bitstrings (ignore the ones that don't change)
    for bitstring in itertools.product(
        ["0", "1"], repeat=num_qubits - (len(control) + len(target))
    ):
        qubits = sorted([(c, "1") for c in control] + [(t, "0") for t in target])

        control_bitstring = list(bitstring)
        target_bitstring = list(bitstring)

        for idx, type_ in qubits:
            control_bitstring.insert(idx, type_)
            target_bitstring.insert(idx, "1")

        # find indices for control/target bitstrings and insert unitary
        idx_0, idx_1 = int("".join(control_bitstring), 2), int("".join(target_bitstring), 2)
        state[idx_0], state[idx_1] = unitary @ [
            state[idx_0],
            state[idx_1],
        ]

    return state
