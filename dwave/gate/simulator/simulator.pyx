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
    "sample_qubit",
]

import random
import warnings
from typing import List, Optional

import numpy as np
cimport numpy as np

import dwave.gate.operations as ops
from dwave.gate.circuit import Circuit, CircuitError

from dwave.gate.simulator.ops cimport (
    apply_cswap,
    apply_gate_control,
    apply_gate_two_control,
    apply_swap,
    measurement_computational_basis,
    single_qubit,
)


def apply_instruction(
    num_qubits: int,
    state: np.ndarray,
    op: ops.Operation,
    targets: list[int],
    little_endian: bool,
    rng: np.random.Generator,
    conjugate_gate: bool = False,
):
    if op.is_blocked:
        return

    if isinstance(op, ops.Measurement):
        if little_endian:
            raise CircuitError("Measurements only supports big-endian qubit indexing.")
        _measure(op, state, targets, rng)

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
    circuit: Circuit,
    mixed_state: bool = False,
    little_endian: bool = False,
    rng_seed: Optional[int] = None,
) -> None:
    """Simulate the given circuit with either a state vector or density matrix simulation.

    The resulting state is stored in the circuit object, together with the measured value in the
    classical register.

    Args:
        circuit: The circuit to simulate.

        mixed_state:
            If true, use the full density matrix method to simulate the circuit.
            Otherwise, simulate using the state vector method.

        little_endian:
            If true, return the state vector using little-endian indexing for
            the qubits. Otherwise use big-endian.

    """

    num_qubits = circuit.num_qubits
    if num_qubits == 0:
        return

    rng = np.random.default_rng(rng_seed)

    if mixed_state:
        state = _simulate_circuit_density_matrix(circuit, rng)
    else:
        state = np.zeros(1 << num_qubits, dtype=np.complex128)
        state[0] = 1

        for op in circuit.circuit:
            targets = [circuit.qubits.index(qb) for qb in op.qubits]
            apply_instruction(num_qubits, state, op, targets, little_endian, rng)

    circuit.set_state(state, force=True)


def sample_qubit(
    qubit: int,
    state: np.typing.NDArray,
    rng: np.random.Generator,
    collapse_state: bool = True,
    little_endian: bool = False,
) -> int:
    """Sample a single qubit.

    Args:
        qubit: The qubit index that is measured.
        state: The state to sample from.
        rng: Random number generator to use for measuring in the computational basis.
        collapse_state: Whether to collapse the state after measuring.
        little_endian: If true, return the state vector using little-endian indexing for
            the qubits. Otherwise use big-endian.

    Returns:
        int: The measurement sample (0 or 1).
    """
    cdef int num_qubits = round(np.log2(state.shape[0]))

    return measurement_computational_basis(
        num_qubits,
        state,
        qubit,
        rng,
        little_endian=little_endian,
        apply_operator=collapse_state,
    )


def _measure(op, state, targets, rng):
    op._measured_state = state.copy()

    for idx, t in enumerate(targets):
        m = sample_qubit(t, state, rng)

        try:
            op.bits[idx].set(m, force=True)
        except (IndexError, TypeError):
            warnings.warn("Measurements not stored in the classical register.")

        op._measured_qubit_indices.append(t)


def _simulate_circuit_density_matrix(
    circuit: Circuit,
    rng: np.random.Generator,
    little_endian: bool = False,
) -> np.typing.NDArray:

    num_qubits = circuit.num_qubits
    num_virtual_qubits = 2 * num_qubits
    state = np.zeros(1 << num_virtual_qubits, dtype=np.complex128)
    state[0] = 1

    for op in circuit.circuit:
        if isinstance(op, ops.Measurement):
            _measure(op, state, circuit, rng)
        else:
            # first apply the gate normally
            targets = [circuit.qubits.index(qb) for qb in op.qubits]
            apply_instruction(
                num_virtual_qubits, state, op, targets, little_endian, rng
            )
            # then apply conjugate transpose to corresponding virtual qubit
            virtual_targets = [t + num_qubits for t in targets]
            apply_instruction(
                num_virtual_qubits, state, op, virtual_targets, little_endian, rng,
                conjugate_gate=True,
            )
    density_matrix = state.reshape((1 << num_qubits, 1 << num_qubits))

    return density_matrix
