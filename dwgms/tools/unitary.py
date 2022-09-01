# Confidential & Proprietary Information: D-Wave Systems Inc.
from __future__ import annotations

import itertools
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from dwgms.operations.base import Operation


#####################
# Circuit functions #
#####################

# TODO: exchange for something better; only here for testing matrix creation
# for custom operations; controlled operations only works with single
# control and target; no support for any other multi-qubit gates
@lru_cache
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
    if op.qubits[0] == qubits[0]:
        mat = op.matrix
    else:
        mat = np.eye(2**op.num_qubits)

    for qb in qubits[1:]:
        if qb == op.qubits[0]:
            mat = np.kron(mat, op.matrix)
        else:
            mat = np.kron(mat, np.eye(2**op.num_qubits))

    return mat @ state


def _apply_controlled_gate(state: NDArray, op: Operation, qubits) -> NDArray:
    """Apply a controlled qubit gate to the state.

    Args:
        state: State on which to apply the operation.
        op: Controlled operation to apply (must have ``op.control``,
            and ``op.target`` defined).
        qubits: List of qubits in the circuit.

    Returns:
        NDArray: Resulting state vector after application.
    """
    control_idx = [qubits.index(c) for c in op.control]
    target_idx = [qubits.index(t) for t in op.target]
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
    dtype: DTypeLike = np.complex,
) -> NDArray:
    """Build the unitary matrix for a controlled operation.

    Args:
        control: Index of control qubit(s).
        target: Index of target qubit. Only a single target supported.
        num_qubits: Total number of qubits.

    Returns:
        NDArray: Unitary matrix representing the controlled operation.
    """
    if not set(control).isdisjoint(target):
        raise ValueError("Control qubits and target qubit cannot be the same.")

    if num_qubits and num_qubits < max(control + target):
        raise ValueError(
            f"Total number of qubits {num_qubits} must be larger or equal "
            f"to the largest qubit index {max(control + target)}."
        )

    if isinstance(control, int):
        control = [control]
    if isinstance(target, int):
        target = [target]

    # TODO: add support for multiple targets
    if len(target) != 1:
        raise NotImplementedError("Multiple target not currently supported.")

    # if None, set number of qubits to the max control/target value
    num_qubits = num_qubits or max(control + target) + 1

    state = np.eye(2**num_qubits, dtype=dtype)

    # create all control and target bitstrings (ignore the ones that don't change)
    for prod in itertools.product(
        ["0", "1"], repeat=num_qubits - (len(control) + len(target))
    ):
        bitstring = "".join(prod)
        qubits = sorted([(c, "1") for c in control] + [(t, "0") for t in target])

        control_bitstring = target_bitstring = ""
        start_idx = 0
        for idx, type_ in qubits:
            control_bitstring += bitstring[start_idx:idx] + type_
            target_bitstring += bitstring[start_idx:idx] + "1"
            start_idx = idx

        # find indices for control/target bitstrings and insert unitary
        idx_0, idx_1 = int(control_bitstring, 2), int(target_bitstring, 2)
        state[idx_0], state[idx_1] = unitary @ [
            state[idx_0],
            state[idx_1],
        ]

    return state