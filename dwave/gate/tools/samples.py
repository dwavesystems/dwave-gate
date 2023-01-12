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
