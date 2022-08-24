# Confidential & Proprietary Information: D-Wave Systems Inc.
import itertools
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray


def build_controlled_unitary(
    control: Union[int, Sequence[int]],
    target: Union[int, Sequence[int]],
    unitary: NDArray,
    num_qubits: Optional[int] = None,
    dtype: DTypeLike = np.complex,
) -> NDArray:
    """Build the unitary matrix for a controlled operation.

    Args:
        control: Control qubit(s).
        target: Target qubit. Only a single target supported.
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
