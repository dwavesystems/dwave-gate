#cython: language_level=3

import numpy as np

from dwgms.simulator.ops cimport (apply_cswap, apply_gate_control,
                                  apply_gate_two_control, apply_swap,
                                  single_qubit)


def apply_instruction(num_qubits, state, op, targets, conjugate_gate=False):
    if op.label == "swap":
        gate = op.matrix
        target0 = targets[1]
        target1 = targets[0]
        apply_swap(num_qubits, state, gate, target0, target1)

    elif op.label == "cswap":
        gate = op.matrix
        target0 = targets[1]
        target1 = targets[2]
        control = targets[0]
        apply_cswap(num_qubits, state, gate, target0, target1, control)

    elif hasattr(op, "control"):
        # should be single qubit gate with controls
        gate = op.target_operation.matrix
        assert gate.shape == (2, 2), (op, gate)
        if conjugate_gate:
            gate = np.ascontiguousarray(gate.conjugate())

        if len(op.control) == 1:
            target = targets[1]
            control = targets[0]
            apply_gate_control(num_qubits, state, gate, target, control)

        elif len(op.control) == 2:
            target = targets[2]
            control0 = targets[0]
            control1 = targets[1]

            apply_gate_two_control(num_qubits, state, gate, target, control0, control1)
    else:
        # apply single qubit gate
        gate = op.matrix
        if conjugate_gate:
            gate = np.ascontiguousarray(gate.conjugate())

        target = targets[0]
        single_qubit(num_qubits, state, gate, target)


def simulate(circuit, mixed_state=False):
    if mixed_state:
        return _simulate_circuit_density_matrix(circuit)

    num_qubits = circuit.num_qubits
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[0] = 1

    for op in circuit.circuit:
        targets = [circuit.qubits.index(qb) for qb in op.qubits]
        apply_instruction(num_qubits, state, op, targets)

    # TODO: workaround to have big-endian (rather that little-endian) output states;
    # otherwise, if big-endian is preferred, just return state as is
    bitstrings = [bin(i)[2:].zfill(num_qubits) for i in range(2 ** num_qubits)]
    swaps = {i: bitstrings.index(bitstrings[i][::-1]) for i in range(2 ** num_qubits)}
    return np.array([state[swaps[i]] for i in range(2 ** num_qubits)])


# TODO: get working with big-endian (rather that little-endian) output states
def _simulate_circuit_density_matrix(circuit):
    num_qubits = circuit.num_qubits
    num_virtual_qubits = 2 * num_qubits
    state = np.zeros(1 << num_virtual_qubits, dtype=np.complex128)
    state[0] = 1

    for op in circuit.circuit:
        # first apply the gate normally
        targets = [circuit.qubits.index(qb) for qb in op.qubits]
        apply_instruction(num_virtual_qubits, state, op, targets)
        # then apply conjugate transpose to corresponding virtual qubit
        virtual_targets = [t + num_qubits for t in targets]
        apply_instruction(num_virtual_qubits, state, op, virtual_targets, conjugate_gate=True)
    density_matrix = state.reshape((1 << num_qubits, 1 << num_qubits))

    return density_matrix
