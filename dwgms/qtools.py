# Confidential & Proprietary Information: D-Wave Systems Inc.
import itertools
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def build_controlled_unitary(
    control: int, target: int, unitary: NDArray, num_qubits: Optional[int] = None
) -> NDArray:
    """Build the unitary matrix for a controlled operation.

    Args:
        control: Control qubit.
        target: Target qubit.
        num_qubits: Total number of qubits.

    Returns:
        NDArray: Unitary matrix representing the controlled operation.
    """
    num_qubits = num_qubits or abs(control - target) + 1
    state = np.eye(2**num_qubits)

    for prod in itertools.product(["0", "1"], repeat=num_qubits - 2):
        bs = "".join(prod)
        is_less = int(target < control)

        tc = control, target
        if is_less:
            tc = target, control

        cbs = f"{bs[: tc[0]]}{abs(is_less - 1)}{bs[tc[0] : tc[1] - 1]}{is_less}{bs[tc[1] - 1 :]}"
        tbs = f"{bs[: tc[0]]}{'1'}{bs[tc[0] : tc[1] - 1]}{'1'}{bs[tc[1] - 1 :]}"

        idx_0, idx_1 = int(cbs, 2), int(tbs, 2)
        state[idx_0], state[idx_1] = unitary @ [
            state[idx_0],
            state[idx_1],
        ]

    return state
