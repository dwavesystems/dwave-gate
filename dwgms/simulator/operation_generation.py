import cgen as c
import numpy as np


def binary(i, n):
    return tuple((i >> j) & 1 for j in range(n))


def compile_gate(gate_matrix, assume_unitary=True):
    gate_matrix_size = len(gate_matrix)

    # set of "sub states" we will need to fetch at each iteration
    sub_states = set()

    # keys are substates, and values are lists of instructions to compute the new
    # sub state amplitude
    instructions = {}

    for state_i in range(gate_matrix_size):
        nz = np.nonzero(gate_matrix[state_i])[0]
        nz_set = set(nz)
        if (
            nz_set == {state_i}
            and gate_matrix[state_i][state_i] == 1
            and assume_unitary
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

    print(gate_matrix)
    print(sub_states, instructions)
    return sub_states, instructions


def generate_op(
    op_name,
    num_targets,
    num_controls,
    sparse_gate_mask=None,
    precompile_gate=None,
    dephasing_epsilon=0,
):
    amplitude_t = "std::complex<double>"

    sub_states = set()
    instructions = {}
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
        print(
            "gate_vals",
            [
                [(amplitude_t, f"gate_value{i}{j}") for j in range(gate_matrix_size)]
                for i in range(gate_matrix_size)
            ],
        )
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

    num_qubits = c.Value("int", "num_qubits")
    state_vector = c.Pointer(c.Value(amplitude_t, "state_vector"))
    targets = [c.Value("int", f"target{i}") for i in range(num_targets)]
    controls = [c.Value("int", f"control{i}") for i in range(num_controls)]

    idx = c.Value("uint64_t", "idx")

    states = {state: c.Value("uint64_t", f"state{state}") for state in sub_states}
    amps = {state: c.Value(amplitude_t, f"amp{state}") for state in sub_states}

    state_idx_mask = c.Value("uint64_t", "state_idx_mask")

    positions = c.ArrayOf(c.Value("uint64_t", "positions"))
    masks = [c.Value("uint64_t", f"mask{i}") for i in range(num_targets + num_controls)]

    num_states = c.Value("uint64_t", "num_states")

    num_iterations = f"({num_states.name} >> {num_targets + num_controls})"

    control_mask = c.Value("uint64_t", "control_mask")
    control_mask_init = " | ".join(
        ["0"] + [f"(1 << {control.name})" for control in controls]
    )

    target_masks = {
        state: c.Value("uint64_t", f"target_mask{state}") for state in sub_states
    }
    target_masks_init = {
        state: " | ".join(
            ["0"]
            + [
                f"({b} << {target.name})"
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

    func = c.FunctionBody(
        c.FunctionDeclaration(c.Value("void", op_name), arguments),
        c.Block(
            [
                c.Initializer(num_states, f"1 << {num_qubits.name}"),
                c.Initializer(
                    positions,
                    "{" + ", ".join([v.name for v in (targets + controls)]) + "}",
                ),
                c.Statement(
                    f"std::sort({positions.name}, {positions.name} + {num_targets + num_controls})"
                ),
                *[
                    c.Initializer(mask, f"(1 << {positions.name}[{i}]) - 1")
                    for i, mask in enumerate(masks)
                ],
                c.Initializer(control_mask, control_mask_init),
                *[
                    c.Initializer(target_masks[st], target_masks_init[st])
                    for st in sorted(sub_states)
                ],
                *inits,
                c.Pragma("omp parallel for"),
                c.For(
                    f"{idx.typename} {idx.name} = 0",
                    f"{idx.name} < {num_iterations}",
                    f"{idx.name}++",
                    c.Block(
                        [
                            # initialize the "state index mask"
                            c.Initializer(state_idx_mask, idx.name),
                            *[
                                c.Assign(
                                    state_idx_mask.name,
                                    f"(({state_idx_mask.name} & ~{mask.name}) << 1) | ({state_idx_mask.name} & {mask.name})",
                                )
                                for mask in masks
                            ],
                            # set all control bits to one in the state mask
                            c.Assign(
                                state_idx_mask.name,
                                f"{state_idx_mask.name} | {control_mask.name}",
                            ),
                            # initialize the states we'll select
                            *[
                                c.Initializer(
                                    states[st],
                                    f"{state_idx_mask.name} | {target_masks[st].name}",
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
            ]
        ),
    )

    # cython_gate_arg = "np.complex128_t* gate_matrix, " if precompile_gate is None else ""
    cython_qubit_args = ", ".join(
        [f"int target{i}" for i in range(num_targets)]
        + [f"int control{i}" for i in range(num_controls)]
    )
    cython_header = f'void c_{op_name} "{op_name}" (int num_qubits, np.complex128_t* state_vector, np.complex128_t* gate_matrix, {cython_qubit_args})'

    # cython_gate_arg = "complex[:,:] gate_matrix, " if precompile_gate is None else ""
    # gate_arg = "gate_matrix"
    qubit_args = ", ".join(
        [f"target{i}" for i in range(num_targets)]
        + [f"control{i}" for i in range(num_controls)]
    )
    cython_function = f"""
cdef inline {op_name}(int num_qubits, complex[::1] state_vector, complex[:,:] gate_matrix, {cython_qubit_args}):
    c_{op_name}(num_qubits, <np.complex128_t*>&state_vector[0], <np.complex128_t*>&gate_matrix[0, 0], {qubit_args})
"""

    return dict(
        function_definition=func,
        cython_header=cython_header,
        cython_function=cython_function,
    )


if __name__ == "__main__":
    c_file = """// THIS FILE WAS AUTOMATICALLY GENERATED BY dwgms/simulator/operation_generation.py
#include <complex.h>
#include <algorithm>


"""

    cython_imports = """
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

    for name, num_targets, num_controls, kwargs in [
        ("single_qubit", 1, 0, {}),
        ("apply_gate_control", 1, 1, {}),
        ("apply_gate_two_control", 1, 2, {}),
        ("apply_swap", 2, 0, dict(precompile_gate=swap_matrix)),
        ("apply_cswap", 2, 1, dict(precompile_gate=swap_matrix)),
        ("apply_dephase_0", 1, 0, dict(precompile_gate=dephase0)),
        ("apply_dephase_1", 1, 0, dict(precompile_gate=dephase1)),
        ("apply_amp_damp_0", 1, 0, dict(precompile_gate=amp_damp0)),
        ("apply_amp_damp_1", 1, 0, dict(precompile_gate=amp_damp1)),
    ]:
        op = generate_op(name, num_targets, num_controls, **kwargs)
        c_file += str(op["function_definition"]) + "\n\n"
        cython_imports += "    " + op["cython_header"] + "\n"
        cython_functions += op["cython_function"] + "\n"

    with open("./dwgms/simulator/ops.h", "w") as f:
        f.write(c_file)

    cython_file = (
        """# THIS FILE WAS AUTOMATICALLY GENERATED BY dwgms/simulator/operation_generation.py
cimport numpy as np
from libc.stdint cimport uint64_t

"""
        + cython_imports
        + "\n"
        + cython_functions
    )

    with open("./dwgms/simulator/ops.pxd", "w") as f:
        f.write(cython_file)
