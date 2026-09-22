# %% [markdown]
# Copyright 2026 D-Wave
#
# The software is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate


def rand_circ_fixed_hamming_weight(
    num_qubits: int,
    hamming_weight: int,
    depth: int,
    seed: int | None = None,
    measure: bool = True,
) -> QuantumCircuit:
    """Generates a quantum circuit with all output bitstrings in
        the ideal distribution having a fixed hamming weight.

    Args:
        num_qubits : Number of qubits
        hamming_weight : Desired hamming weight of ideal outputs
        depth: Circuit depth
        seed : Seed for RNG. Defaults to None.
        measure : Add measurements to end of circuit. Defaults to True.

    Raises:
        ValueError: Invalid hamming weight
        ValueError: Invalid depth

    Returns:
        QuantumCircuit: Random quantum circuit outputting a state
        with the hamming weight specified
    """
    if not 0 <= hamming_weight <= num_qubits:
        raise ValueError("hamming_weight must be between 0 and num_qubits")

    if depth < 0:
        raise ValueError("depth must be nonnegative")

    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits)

    # Random initial basis state with exactly hamming_weight ones.
    occupied = rng.choice(
        num_qubits,
        size=hamming_weight,
        replace=False,
    )

    for qubit in occupied:
        circuit.x(int(qubit))

    for _ in range(depth):
        gate_type = rng.choice(["rz", "cz", "xy", "swap"])

        if gate_type == "rz":
            qubit = int(rng.integers(num_qubits))
            angle = rng.uniform(0.0, 2.0 * np.pi)
            circuit.rz(angle, qubit)

        else:
            q0, q1 = rng.choice(
                num_qubits,
                size=2,
                replace=False,
            )
            q0 = int(q0)
            q1 = int(q1)

            if gate_type == "cz":
                circuit.cz(q0, q1)

            elif gate_type == "swap":
                circuit.swap(q0, q1)

            elif gate_type == "xy":
                theta = rng.uniform(0.0, 2.0 * np.pi)
                beta = rng.uniform(0.0, 2.0 * np.pi)

                circuit.append(
                    XXPlusYYGate(theta, beta),
                    [q0, q1],
                )

    if measure:
        circuit.measure_all()

    return circuit
