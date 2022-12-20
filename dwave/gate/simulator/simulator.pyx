# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

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

__all__ = [
    "simulate",
]

import random
import warnings
from typing import List

import numpy as np

import dwave.gate.operations as ops
from dwave.gate.circuit import Circuit

from dwave.gate.simulator.ops cimport (
    apply_cswap,
    apply_gate_control,
    apply_gate_two_control,
    apply_swap,
    single_qubit,
)


def apply_instruction(
    num_qubits: int,
    state: np.ndarray,
    op: ops.Operation,
    targets: list[int],
    little_endian: bool,
    conjugate_gate: bool = False,
):
    # if conditional and all bits are 1, skip
    if op._cond and not any(op._cond):
        return

    elif isinstance(op, ops.SWAP):
        gate = op.matrix
        target0 = targets[1]
        target1 = targets[0]
        apply_swap(
            num_qubits, state, gate, target0, target1,
            little_endian=little_endian,
        )

    elif isinstance(op, ops.CSWAP):
        gate = op.matrix
        target0 = targets[1]
        target1 = targets[2]
        control = targets[0]
        apply_cswap(
            num_qubits, state, gate, target0, target1, control,
            little_endian=little_endian,
        )

    elif isinstance(op, ops.CCX):
        # this one has to hardcoded for now because ops.ControlledOperation
        # doesn't support more than one control yet
        gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        target = targets[2]
        control0 = targets[0]
        control1 = targets[1]
        apply_gate_two_control(
            num_qubits, state, gate, target, control0, control1,
            little_endian=little_endian,
        )

    elif isinstance(op, ops.ControlledOperation):
        # should be single qubit gate with controls
        if isinstance(op, ops.ParametricOperation):
            gate = op.target_operation(op.parameters).matrix
        else:
            gate = op.target_operation.matrix
        assert gate.shape == (2, 2), (op, gate)
        if conjugate_gate:
            gate = np.ascontiguousarray(gate.conjugate())

        if op.num_control == 1:
            target = targets[1]
            control = targets[0]
            apply_gate_control(
                num_qubits, state, gate, target, control,
                little_endian=little_endian,
            )

        elif op.num_control == 2:
            target = targets[2]
            control0 = targets[0]
            control1 = targets[1]

            apply_gate_two_control(
                num_qubits, state, gate, target, control0, control1,
                little_endian=little_endian,
            )
    else:
        # apply single qubit gate
        if op.num_qubits != 1:
            raise ValueError(
                f"simulator encountered unknown multi-qubit operation: {op.label}"
            )
        gate = op.matrix
        if conjugate_gate:
            gate = np.ascontiguousarray(gate.conjugate())

        target = targets[0]
        single_qubit(num_qubits, state, gate, target, little_endian=little_endian)


def simulate(
    circuit: Circuit, mixed_state: bool = False, little_endian: bool = False
) -> np.typing.NDArray:
    """Simulate the given circuit with either a state vector or density matrix
    simulation.

    Args:
        circuit: The circuit to simulate.

        mixed_state:
            If true, use the full density matrix method to simulate the circuit.
            Otherwise, simulate using the state vector method.

        little_endian:
            If true, return the state vector using little-endian indexing for
            the qubits. Otherwise use big-endian.

    Returns:
        The resulting state vector or density matrix.

    """
    num_qubits = circuit.num_qubits
    if num_qubits == 0:
        return np.empty(0, dtype=np.complex128)

    if mixed_state:
        return _simulate_circuit_density_matrix(circuit)

    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[0] = 1

    for op in circuit.circuit:
        if isinstance(op, ops.Measurement):
            _measure(op, state, circuit)
        else:
            targets = [circuit.qubits.index(qb) for qb in op.qubits]
            apply_instruction(num_qubits, state, op, targets, little_endian)

    return state


def _measure(op, state, circuit):
    op._measured_state = state.copy()

    for idx, qb in enumerate(op.qubits):
        m = _sample(circuit.qubits.index(qb), state)

        try:
            op.bits[idx].set(m)
        except (IndexError, TypeError):
            warnings.warn("Measurements not stored in the classical register.")

        op._measured_qubit_indices.append(circuit.qubits.index(qb))


def _sample(qubit: int, state: np.typing.NDArray, collapse_state: bool = True) -> int:
    """Sample a measurement on a single qubit and collapse the state.

    Args:
        qubit: Which qubit (index, from left to right) to sample.
        collapse_state: Whether to collapse the state after a measurement.

    Returns:
        int: 0 or 1 sample of the measured qubit.
    """
    num_qubits = round(np.log2(len(state)))

    if qubit >= num_qubits:
        raise ValueError(f"Cannot sample qubit {qubit}. State has only {num_qubits} qubits.")

    weight_0 = weight_1 = 0
    zero_weight = False
    state_0_ids = []
    state_1_ids = []

    for i in range(2**num_qubits):
        if i % 2**(num_qubits - qubit - 1) == 0:
            zero_weight = not zero_weight
        if zero_weight:
            weight_0 += np.abs(state[i])**2
            state_0_ids.append(i)
        else:
            weight_1 += np.abs(state[i])**2
            state_1_ids.append(i)

    sample = random.choices((0, 1), weights=[weight_0, weight_1])[0]

    if collapse_state:
        if sample == 0:
            state[state_1_ids] = 0
        else:
            state[state_0_ids] = 0

        # renormalize the state after removing the measured state values
        norm = np.linalg.norm(state)
        for i, s in enumerate(state):
            state[i] = state[i] / norm

    return sample


def _simulate_circuit_density_matrix(circuit: Circuit, little_endian: bool = False) -> np.typing.NDArray:
    num_qubits = circuit.num_qubits
    num_virtual_qubits = 2 * num_qubits
    state = np.zeros(1 << num_virtual_qubits, dtype=np.complex128)
    state[0] = 1

    for op in circuit.circuit:
        # first apply the gate normally
        targets = [circuit.qubits.index(qb) for qb in op.qubits]
        apply_instruction(
            num_virtual_qubits, state, op, targets, little_endian
        )
        # then apply conjugate transpose to corresponding virtual qubit
        virtual_targets = [t + num_qubits for t in targets]
        apply_instruction(
            num_virtual_qubits, state, op, virtual_targets, little_endian,
            conjugate_gate=True,
        )
    density_matrix = state.reshape((1 << num_qubits, 1 << num_qubits))

    return density_matrix
