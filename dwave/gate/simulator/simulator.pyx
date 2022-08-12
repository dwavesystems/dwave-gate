#cython: language_level=3

cimport numpy as np
from libc.stdint cimport uint64_t
from cython.parallel import prange, threadid
from libc.stdlib cimport abort, malloc, free
cimport cython
cimport openmp

import numpy as np

from dwave.gate.simulator.ops cimport (
    single_qubit,
    apply_gate_control,
    apply_gate_two_control,
    apply_swap,
    apply_cswap,

    apply_dephase_0,
    apply_dephase_1,

    apply_amp_damp_0,
    apply_amp_damp_1,
)


def py_apply_dephase_0(num_qubits, state, target):
    apply_dephase_0(num_qubits, state, np.zeros((2, 2), dtype=np.complex128), target)


def py_apply_dephase_1(num_qubits, state, target):
    apply_dephase_1(num_qubits, state, np.zeros((2, 2), dtype=np.complex128), target)


def py_apply_amp_damp_0(num_qubits, state, target):
    apply_amp_damp_0(num_qubits, state, np.zeros((2, 2), dtype=np.complex128), target)


def py_apply_amp_damp_1(num_qubits, state, target):
    apply_amp_damp_1(num_qubits, state, np.zeros((2, 2), dtype=np.complex128), target)


def apply_kraus(num_qubits, state, temp_state, target, operators):
    """Apply a Kraus operator chosen randomly from the given set, normalized
    for probability.

    See https://confluence.dwavesys.com/display/AD/Simulating+noise+with+Kraus+operators
    """

    u = np.random.uniform()
    prob_sum = 0

    for op in operators[:-1]:
        # copy the state over temporarily and apply the operator
        temp_state[:] = state
        op(num_qubits, temp_state, target)

        # calculate the relative probability
        prob = np.sqrt(temp_state.dot(temp_state.conjugate()).real)

        prob_sum += prob
        if prob_sum > u:
            state[:] = temp_state / prob
            break
    else:
        # only one operator left, it must be applied
        operators[-1](num_qubits, state, target)


def apply_instruction(num_qubits, state, instruction, targets, conjugate_gate=False):
    if instruction.name == "swap":
        gate = instruction.to_matrix()
        target0 = targets[1]
        target1 = targets[0]
        apply_swap(num_qubits, state, gate, target0, target1)

    elif instruction.name == "cswap":
        gate = instruction.to_matrix()
        target0 = targets[1]
        target1 = targets[2]
        control = targets[0]
        apply_cswap(num_qubits, state, gate, target0, target1, control)

    elif hasattr(instruction, "num_ctrl_qubits"):
        # should be single qubit gate with controls
        gate = instruction.base_gate.to_matrix()
        assert gate.shape == (2, 2), (instruction, gate)
        if conjugate_gate:
            gate = np.ascontiguousarray(gate.conjugate())

        if instruction.num_ctrl_qubits == 1:
            target = targets[1]
            control = targets[0]
            apply_gate_control(num_qubits, state, gate, target, control)

        elif instruction.num_ctrl_qubits == 2:
            target = targets[2]
            control0 = targets[0]
            control1 = targets[1]

            apply_gate_two_control(num_qubits, state, gate, target, control0, control1)
    else:
        # single qubit gate
        gate = instruction.to_matrix()
        if conjugate_gate:
            gate = np.ascontiguousarray(gate.conjugate())

        target = targets[0]
        single_qubit(num_qubits, state, gate, target)


def simulate_circuit(circuit):
    num_qubits = circuit.num_qubits
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[0] = 1

    for ins, qubits, _ in circuit.data:
        targets = [q.index for q in qubits]
        apply_instruction(num_qubits, state, ins, targets)

    return state


def simulate_circuit_with_kraus_errors(
    circuit,
    apply_dephase=False,
    apply_amplitude_damping=False
):
    num_qubits = circuit.num_qubits
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[0] = 1

    # this will be used as a placeholder when iterating over possible kraus operators
    state_copy = np.empty(1 << num_qubits, dtype=np.complex128)

    for ins, qubits, _ in circuit.data:
        targets = [q.index for q in qubits]
        apply_instruction(num_qubits, state, ins, targets)

        # apply dephase and/or amplitude damping to each qubit affected by the gate
        # no idea if this is a reasonable way to do it, but it serves as a
        # proof of concept
        if apply_dephase:
            for target in targets:
                apply_kraus(
                    num_qubits, state, state_copy, target,
                    [py_apply_dephase_0, py_apply_dephase_1]
                )

        if apply_amplitude_damping:
            for target in targets:
                apply_kraus(
                    num_qubits, state, state_copy, target,
                    [py_apply_amp_damp_0, py_apply_amp_damp_1]
                )

    return state


def simulate_circuit_density_matrix(circuit):
    num_qubits = circuit.num_qubits
    num_virtual_qubits = 2 * num_qubits
    state = np.zeros(1 << num_virtual_qubits, dtype=np.complex128)
    state[0] = 1

    for ins, qubits, _ in circuit.data:
        # first apply the gate normally
        targets = [q.index for q in qubits]
        apply_instruction(num_virtual_qubits, state, ins, targets)
        # then apply conjugate transpose to corresponding virtual qubit
        virtual_targets = [t + num_qubits for t in targets]
        apply_instruction(num_virtual_qubits, state, ins, virtual_targets, conjugate_gate=True)
    density_matrix = state.reshape((1 << num_qubits, 1 << num_qubits))

    return density_matrix
