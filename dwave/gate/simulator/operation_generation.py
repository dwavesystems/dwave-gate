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

from typing import List, Dict, Tuple, Optional, NamedTuple, Union

import cgen as c
import numpy as np


# entry of a gate matrix
EntryType = Union[float, c.Value]


def binary(i: int, n: int) -> Tuple[int, ...]:
    """Return the bitstring of i (with n total bits) as a tuple."""
    return tuple((i >> j) & 1 for j in range(n))


def compile_gate(
    gate_matrix: Union[np.ndarray, list[list[EntryType]]]
) -> Tuple[set[int], dict[int, List[Tuple[int, EntryType]]]]:
    """Compile a given gate into a minimal set of instructions.

    Args:
        gate_matrix: Square matrix representing the gate. Values may be scalars or C variables that
        refer to floats.

    Returns:
        A tuple where the first item is the set of substates acted on by the gate, and the second
        item is a dictionary mapping output substates to those corresponding instructions.
    """

    gate_matrix_size = len(gate_matrix)

    # set of "sub states" we will need to fetch at each iteration
    sub_states = set()

    # keys are substates, and values are lists of instructions to compute the new
    # sub state amplitude
    instructions: Dict[int, List[Tuple[int, EntryType]]] = {}

    for state_i in range(gate_matrix_size):
        nz = np.nonzero(gate_matrix[state_i])[0]
        nz_set = set(nz)
        if (
            nz_set == {state_i}
            and gate_matrix[state_i][state_i] == 1
        ):
            # essentially the identity operation for this sub state, can ignore
            continue

        # sub_states.update(nz_set)
        sub_states.add(state_i)

        instructions[state_i] = []
        for state_j in nz:
            entry = gate_matrix[state_i][state_j]
            if entry != 0:
                sub_states.add(state_j)
                instructions[state_i].append((state_j, entry))

    return sub_states, instructions


class CFuncInfo(NamedTuple):
    function_definition: str
    cython_qubit_args: str
    cython_header: str


class OpInfo(NamedTuple):
    function_definition: str
    cython_header: str
    cython_function: str


def generate_op_c_code(
    op_name: str,
    num_targets: int,
    num_controls: int,
    sparse_gate_mask: Optional[np.ndarray] = None,
    precompile_gate: Optional[np.ndarray] = None,
    little_endian: bool = True,
) -> CFuncInfo:
    """Generate the C code (and some cython headers) that implements a gate in
    an optimal way.

    The main algorithm is to iterate (in parallel) over all basis states of the qubits
    unaffected by the gate. At each iteration, create a "basis template", essentially
    a bit string of the basis state with gaps for each of the affected qubits. Then,
    depending on if the affected qubits are targets or controls, the full basis states
    that we need to select to do the matrix operation are generated.

    This is done with bitwise operations. We start by looping from 0 to
    2^(num unaffected qubits). The iteration variable (partial_basis) is equal to the
    basis state of all unaffected qubits (but with no gaps for affected qubits in the
    bits), and then for each affected qubit shifting the upper bits over one and ORing
    back in the lower bits.

    Example: 5 qubits, gate affects qubits 1 and 3, and partial_basis is 6

    mask0 = 0b00001 is for qubit 1, mask1 = 0b00111 is for qubit 3

    start with basis_template = partial_basis = 6 = 0b00110

    we want to create a basis_template `1x1x0` (where xs will be set
    to zero so we can fill in later)

    first iteration:
       basis_template = ((basis_template & ~mask0) << 1) | (basis_template & mask0)
                      = ((0b00110 & 0b11110) << 1) | (0b00110 & 0b0001)
                      = 0b01100 | 0b00000
                      = 0b01100

    second iteration:
       basis_template = ((basis_template & ~mask1) << 1) | (basis_template & mask1)
                      = ((0b01100 & 0b11000) << 1) | (0b01100 & 0b00111)
                      = 0b10000 | 0b00100
                      = 0b10100

    now basis_template can be used to generate all the indices needed
    to apply the gate while holding unaffected qubits fixed to one
    basis state.

    Args:
        op_name:
            A name for the gate which will be used in the names of the C functions.

        num_targets: Number of targets (non-control qubits) of the gate.

        num_controls: Number of controls in the gate.

        sparse_gate_mask:
            A 2-d binary matrix. If the entry is 0, the entry will not be used when
            applying this gate. This can improve performance, as some
            multiplications/adds can be skipped.

        precompile_gate:
            A 2-d gate matrix. If supplied, this values of the matrix will be inserted
            into the C code, which may save time when doing the matrix multiplication.

        little_endian:
            If true, use little-endian convention for the basis states. Otherwise, use
            big-endian.

    Returns:
        C code/cython headers that implement the gate.

    """

    amplitude_t = "std::complex<double>"

    num_qubits = c.Value("uint64_t", "num_qubits")

    def qubit_index(qubit):
        if not little_endian:
            return f"({num_qubits.name} - {qubit} - 1)"
        return qubit

    def shift(value, idx):
        return f"({value} << {idx})"

    sub_states: set[int] = set()
    inits = []

    gate_matrix_size = 1 << num_targets

    gate_matrix = c.Pointer(c.Value(amplitude_t, "gate_matrix"))

    if sparse_gate_mask is None:
        sparse_gate_mask = np.ones((gate_matrix_size, gate_matrix_size))

    if precompile_gate is not None:
        assert precompile_gate.shape == (gate_matrix_size, gate_matrix_size)
        sub_states, instructions = compile_gate(precompile_gate)
    else:
        gate_values = [
            [c.Value(amplitude_t, f"gate_value{i}{j}") for j in range(gate_matrix_size)]
            for i in range(gate_matrix_size)
        ]
        inits.extend(
            [
                c.Initializer(
                    gate_values[i][j], f"{gate_matrix.name}[{i * gate_matrix_size + j}]"
                )
                for i in range(gate_matrix_size)
                for j in range(gate_matrix_size)
                if sparse_gate_mask[i][j]
            ]
        )
        sub_states, instructions = compile_gate(
            [
                [
                    gate_values[i][j].name if sparse_gate_mask[i][j] else 0
                    for j in range(gate_matrix_size)
                ]
                for i in range(gate_matrix_size)
            ]
        )

    state_vector = c.Pointer(c.Value(amplitude_t, "state_vector"))
    targets = [c.Value("uint64_t", f"target{i}") for i in range(num_targets)]
    controls = [c.Value("uint64_t", f"control{i}") for i in range(num_controls)]

    partial_basis = c.Value("uint64_t", "partial_basis")

    states = {state: c.Value("uint64_t", f"state{state}") for state in sub_states}
    amps = {state: c.Value(amplitude_t, f"amp{state}") for state in sub_states}

    basis_template = c.Value("uint64_t", "basis_template")

    positions = c.ArrayOf(c.Value("uint64_t", "positions"))
    masks = [c.Value("uint64_t", f"mask{i}") for i in range(num_targets + num_controls)]

    num_states = c.Value("uint64_t", "num_states")

    num_iterations = f"({num_states.name} >> {num_targets + num_controls})"

    control_mask = c.Value("uint64_t", "control_mask")
    control_mask_init = " | ".join(
        ["0"] + [shift(1, qubit_index(control.name)) for control in controls]
    )

    target_masks = {
        state: c.Value("uint64_t", f"target_mask{state}") for state in sub_states
    }
    target_masks_init = {
        state: " | ".join(
            ["0"]
            + [
                shift(b, qubit_index(target.name))
                for b, target in zip(binary(state, num_targets), targets)
            ]
        )
        for state in sub_states
    }

    def multiply_val_and_amps(state_i):
        ins = instructions[state_i]
        terms = []

        for state_j, val in ins:
            if val == 1:
                terms.append(amps[state_j].name)
            elif val != 0:
                terms.append(f"({val} * {amps[state_j].name})")

        if len(terms) == 0:
            return "0.0"
        return " + ".join(terms)

    arguments = [num_qubits, state_vector]
    arguments.append(gate_matrix)
    arguments.extend(targets)
    arguments.extend(controls)

    body = c.Block([
        # number of total states
        c.Initializer(num_states, f"1 << {num_qubits.name}"),

        # this array will hold every "position" of the qubits affected by the gate in
        # sorted order
        c.Initializer(
            positions,
            "{" + ", ".join([
                qubit_index(v.name) for v in (targets + controls)
            ]) + "}",
        ),
        c.Statement(
            "std::sort("
            f"{positions.name}, {positions.name} + {num_targets + num_controls}"
            ")"
        ),

        # initialize all of the "masks". these will be used to generate the correct
        # state index
        *[
            c.Initializer(mask, f"(1 << {positions.name}[{i}]) - 1")
            for i, mask in enumerate(masks)
        ],
        c.Initializer(control_mask, control_mask_init),
        *[
            c.Initializer(target_masks[st], target_masks_init[st])
            for st in sorted(sub_states)
        ],

        # rest of the variables we need to initialize
        *inits,


        # loop over all basis states of all qubits *not* affected by the gate
        # c.Pragma("omp parallel for"),
        c.For(
            f"{partial_basis.typename} {partial_basis.name} = 0",
            f"{partial_basis.name} < {num_iterations}",
            f"{partial_basis.name}++",
            c.Block(
                [
                    # initialize the "basis template". see docstring for details
                    c.Initializer(basis_template, partial_basis.name),
                    *[
                        c.Assign(
                            basis_template.name,
                            (
                                f"(({basis_template.name} & ~{mask.name}) << 1) | "
                                f"({basis_template.name} & {mask.name})"
                            )
                        )
                        for mask in masks
                    ],

                    # set all control bits to one in the state mask
                    c.Assign(
                        basis_template.name,
                        f"{basis_template.name} | {control_mask.name}",
                    ),

                    # initialize the states we'll select
                    *[
                        c.Initializer(
                            states[st],
                            f"{basis_template.name} | {target_masks[st].name}",
                        )
                        for st in sorted(sub_states)
                    ],

                    # get their amplitudes
                    *[
                        c.Initializer(
                            amps[st], f"{state_vector.name}[{states[st].name}]"
                        )
                        for st in sorted(sub_states)
                    ],

                    # perform the matrix multiplication
                    *[
                        c.Assign(
                            f"{state_vector.name}[{states[state_i].name}]",
                            multiply_val_and_amps(state_i),
                        )
                        for state_i in sorted(sub_states)
                    ],
                ]
            ),
        ),
    ])

    func = c.FunctionBody(
        c.FunctionDeclaration(c.Value("void", op_name), arguments), body,
    )

    cython_qubit_args = ", ".join(
        [f"uint64_t target{i}" for i in range(num_targets)]
        + [f"uint64_t control{i}" for i in range(num_controls)]
    )
    cython_header = f"""\
    void c_{op_name} "{op_name}" (
        uint64_t num_qubits,
        np.complex128_t* state_vector,
        np.complex128_t* gate_matrix,
        {cython_qubit_args}
    )"""

    return CFuncInfo(func, cython_qubit_args, cython_header)


def generate_op(
    op_name: str,
    num_targets: int,
    num_controls: int,
    sparse_gate_mask: Optional[np.ndarray] = None,
    precompile_gate: Optional[np.ndarray] = None,
) -> OpInfo:
    """Generate C code with cython wrapper for a fast implementation of the given gate
    specification. See :func:`generate_op_c_code` for more detail.

    This will generate both little and big-endian versions of the gate.

    Args:
        op_name:
            A name for the gate which will be used in the names of the C functions.

        num_targets: Number of targets (non-control qubits) of the gate.

        num_controls: Number of controls in the gate.

        sparse_gate_mask:
            A 2-d binary matrix. If the entry is 0, the entry will not be used when
            applying this gate. This can improve performance, as some
            multiplications/adds can be skipped.

        precompile_gate:
            A 2-d gate matrix. If supplied, this values of the matrix will be inserted
            into the C code, which may save time when doing the matrix multiplication.

        little_endian:
            If true, use little-endian convention for the basis states. Otherwise, use
            big-endian.

    Returns:
        C code/cython wrapper that implement the gate.

    """

    c_codes = []
    op_options = [
        (True, f"{op_name}_little_endian"),
        (False, f"{op_name}_big_endian"),
    ]
    for little_endian, full_op_name in op_options:
        c_codes.append(generate_op_c_code(
            full_op_name,
            num_targets,
            num_controls,
            sparse_gate_mask=sparse_gate_mask,
            precompile_gate=precompile_gate,
            little_endian=little_endian,
        ))

    qubit_args = ", ".join(
        [f"target{i}" for i in range(num_targets)]
        + [f"control{i}" for i in range(num_controls)]
    )

    cython_function = f"""\
cdef inline {op_name}(
    uint64_t num_qubits,
    complex[::1] state_vector,
    complex[:,:] gate_matrix,
    {c_codes[0].cython_qubit_args},
    little_endian=True,
):
    if little_endian:
        c_{op_options[0][1]}(
            num_qubits,
            <np.complex128_t*>&state_vector[0],
            <np.complex128_t*>&gate_matrix[0, 0],
            {qubit_args}
        )
    else:
        c_{op_options[1][1]}(
            num_qubits,
            <np.complex128_t*>&state_vector[0],
            <np.complex128_t*>&gate_matrix[0, 0],
            {qubit_args}
        )
"""

    return OpInfo(
        "\n\n".join(
            str(c_code.function_definition) for c_code in c_codes
        ),
        "\n\n".join(c_code.cython_header for c_code in c_codes),
        cython_function,
    )


if __name__ == "__main__":
    c_file = """\
// THIS FILE WAS AUTOMATICALLY GENERATED BY dwave/gate/simulator/operation_generation.py
#include <complex.h>
#include <algorithm>


"""

    cython_imports = """\
cdef extern from "./ops.h" nogil:
"""

    cython_functions = ""

    swap_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    eps = 0.01

    dephase0 = np.array([[np.sqrt(1 - eps), 0], [0, np.sqrt(1 - eps)]])
    dephase1 = np.array([[np.sqrt(eps), 0], [0, -np.sqrt(eps)]])

    amp_damp0 = np.array([[1, 0], [0, np.sqrt(1 - eps)]])
    amp_damp1 = np.array([[0, np.sqrt(eps)], [0, 0]])

    gate_defintions: List[Tuple[str, int, int, Dict]] = [
        ("single_qubit", 1, 0, {}),
        ("apply_gate_control", 1, 1, {}),
        ("apply_gate_two_control", 1, 2, {}),
        ("apply_swap", 2, 0, dict(precompile_gate=swap_matrix)),
        ("apply_cswap", 2, 1, dict(precompile_gate=swap_matrix)),
        ("apply_dephase_0", 1, 0, dict(precompile_gate=dephase0)),
        ("apply_dephase_1", 1, 0, dict(precompile_gate=dephase1)),
        ("apply_amp_damp_0", 1, 0, dict(precompile_gate=amp_damp0)),
        ("apply_amp_damp_1", 1, 0, dict(precompile_gate=amp_damp1)),
    ]

    for name, num_targets, num_controls, kwargs in gate_defintions:
        op = generate_op(name, num_targets, num_controls, **kwargs)
        c_file += str(op.function_definition) + "\n\n"
        cython_imports += op.cython_header + "\n\n"
        cython_functions += op.cython_function + "\n"

    with open("./dwave/gate/simulator/ops.h", "w") as f:
        f.write(c_file)

    cython_file = (
        """\
# THIS FILE WAS AUTOMATICALLY GENERATED BY dwave/gate/simulator/operation_generation.py
# cython: language_level=3

cimport numpy as np
from libc.stdint cimport uint64_t

"""
        + cython_imports
        + "\n"
        + cython_functions
    )

    with open("./dwave/gate/simulator/ops.pxd", "w") as f:
        f.write(cython_file)
