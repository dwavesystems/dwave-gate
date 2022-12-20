import random
from typing import List

import numpy as np
from numpy.typing import NDArray


def sample(qubit: int, state: NDArray, num_samples: int = 1) -> List[int]:
    """Sample a measurement on a single qubit.

    Args:
        qubit: Which qubit (index, from left to right) to sample.
        num_samples: How many samples to return.

    Returns:
        list[int]: 0 or 1 sample(s) of the measured qubit.
    """
    num_qubits = round(np.log2(len(state)))

    if qubit >= num_qubits:
        raise ValueError(f"Cannot sample qubit {qubit}. State has only {num_qubits} qubits.")

    zero_weight = False
    weight_0 = weight_1 = 0
    for i in range(2**num_qubits):
        if i % 2 ** (num_qubits - qubit - 1) == 0:
            zero_weight = not zero_weight
        if zero_weight:
            weight_0 += np.abs(state[i]) ** 2
        else:
            weight_1 += np.abs(state[i]) ** 2

    return random.choices((0, 1), weights=[weight_0, weight_1], k=num_samples)
