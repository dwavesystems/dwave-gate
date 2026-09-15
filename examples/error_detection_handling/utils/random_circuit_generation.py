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
