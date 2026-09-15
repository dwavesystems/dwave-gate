# %% [markdown]
# # Real-time Arithmetic on Classical Registers
# In this notebook, we will show how to implement real-time arithmetic on classical registers attached to the dual-rail qubits, which is a important component for real-time control flow and necessary for error correction.

# %%
import functools
import logging

import numpy as np
import polars as pl

from dwave.gate.qcdl import Scope, qcdl
from dwave.gate.qcdl.constants import LogicalOutcomeToInteger
from dwave.gate.qcdl.operations import h, measure, ry
from dwave.gate.utils.display import print_qcdl

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## 5-qubit Example
# We will prepare a circuit with $n$ `controlling_qubits` and one `main_qubit`. We first apply a set of gates on the `controlling_qubits` and measure them. Denote the measurement outcome as $x_0, x_1, .., x_{n-1}$, we then apply a $R_y(\theta)$ gate on the `main_qubit` with angle
# $$
# \theta = \pi \sum_{j=0}^{n-1} x_j2^{-j-1}
# $$
#
# We start with a simple case of 5 qubits. q0 to q3 are the `controlling_qubits` and q4 is the `main_qubit`. We will apply Hadamard gates on `controlling_qubits` and measure them. Depending on the outcomes, we apply the $R_y(\theta)$ gate on q4.
#
# This program is highly non-trivial to implement and the key steps will be explained.

# %%
# Specify the number of qubits in the QCDL program
num_c_qubits = 5


@qcdl(num_c_qubits)
def main(**kwargs):

    qubits = list(kwargs.values())

    sc = Scope(*qubits)  # Convenience method to utilize qubits as a group

    c_sum = sc.FixedPointRegister(
        0, name="cregs"
    )  # Use FixedPointRegister for decimal values.
    # This initialization will only rest the value before all the shots, but not every shot
    c_sum <<= 0  # We need to explicitly reset the register to 0 here, otherwise it will carry the results of previous shots

    cregs = []
    for q_id, qubit in enumerate(qubits[:-1]):
        cregs.append(sc.Register(0, name=f"creg{q_id}"))
        h(qubit)
        measure(qubit, register=cregs[q_id])  # Measure and store result
        # Store the measurement results for convenient post-processing
        sc.append_table_row(cregs[q_id], table_name=f"q{q_id}_measurement")
        with sc.If(cregs[q_id] == 1):  # when the qubit is measured to be
            # Add 2^(-1-j) to the c_sum register of all the qubits
            c_sum += 2 ** (-1 - q_id)

    sc.append_table_row(c_sum, table_name="sum")
    sc.sync()  # Synchronize all the qubits before applying the gate on the main qubit

    # Note that c_sum doesn't carry the factor of pi, but as it is a classical register,
    # when being passed as a argument of a gate, QCDL assumes it is in the unit of pi, so no need
    # to multiply the factor of pi here.
    # For example, if c_sum==0.5, then the rotation angle here is 0.5pi in radiant
    ry(qubits[-1], c_sum)

    cm = sc.Register(0, name="creg_main")
    measure(qubits[-1], register=cm)
    # Store the measurement results for convenient post-processing
    sc.append_table_row(cm, table_name="qmain_measurement")


# Method to print out the QCDL for verification
# print_qcdl(main())
# If your notebook doesn't display `print_qcdl` output well, try uncommenting the next line instead.
print(print_qcdl(main(), to_Display=False))

# %% [markdown]
# For simulation, we simulate the noiseless case to ensure our implementation is effective. Without circuit noise, the empirical rotation angle should be close to the ideal one, and the difference is solely due to shot noise.

# %%
from dwave.gate.leap import LeapQCDLSimulator

shots = 10000  # Use a large number of shots to reduce shot noise

simulator = LeapQCDLSimulator()
future = simulator.run(main(), shots=shots, noise_model=False)
results = future.result().result

# %%
results.get_counts()


# %%
def print_ry_angle(controlling_states: list, data: pl.DataFrame):
    """Helper function to print the empirical y rotation angle,
    excluding the leakage state

    Args:
        controlling_states (list): list representation
        of the controlling state
        data (pd.DataFrame): measurement data
    """
    print(
        "The state of the controlling qubits is |{}>".format(
            "".join(map(str, controlling_states))
        )
    )
    expr = functools.reduce(
        lambda a, b: a & b,
        [
            (pl.col(f"creg{qubit}") == state)
            for qubit, state in enumerate(controlling_states)
        ]
        + [pl.col("creg_main") != LogicalOutcomeToInteger.SPLAT.value],
    )
    sub_df = df.filter(expr)

    ideal_angle = sub_df.item(row=0, column="cregs")
    all_close = np.all(
        np.isclose(sub_df["cregs"], ideal_angle)
    )  # Validation to ensure all the sums are close
    if all_close:
        print(f"The ideal rotation angle is {ideal_angle:.4f} pi")
    else:
        raise ValueError("Some of the rotation angles have wong value!")
    frac_of_1 = sub_df["creg_main"].mean()
    theta = np.arcsin(np.sqrt(frac_of_1)) * 2 / np.pi
    print(f"The empirical y rotation angle is {theta:.4f} pi")
    print("\n")


# %% [markdown]
# All the classical register data collected with `Scope.append_table_row()` are stored the `records` dictionary returned by `results.get_records()`. For accessing the relevant data, we first need to specify the qubit number. We choose consistent qubit number for the measurement result. (i.e. use `records["q2"]["q2_measurement"]` for the results of `q2`). The main qubit is denoted as `qmain`. The sum is stored in the registers of all the qubits and we choose to access it through `qmain` here (but we can use any qubit for it).
#
# The data of each classical register is a `dictionary` and the key is the name specified at its initialization (e.g. `creg2`). It can be conveniently converted to a `DataFrame` for analysis.

# %%
# Generate a dataframe for the relevant results
records = results.records
df = pl.concat(
    [
        records[f"q{num_c_qubits - 1}"]["qmain_measurement"],
        records[f"q{num_c_qubits - 1}"]["sum"],
    ]
    + [records[f"q{q_id}"][f"q{q_id}_measurement"] for q_id in range(num_c_qubits - 1)],
    how="horizontal_extend",
)
print(df)


# %%
def print_n_bit_angles(num_c_qubits: int, c_st_list: list, ii: int):
    """Recursive function to iterate over all states

    Args:
        num_c_qubits (int): number of controlling qubits
        c_st_list (list): list representation of the controlling state
        ii (int): qubit index of the current function call
    """
    if ii == num_c_qubits:
        print_ry_angle(c_st_list, df)
        return

    c_st_list[ii] = 0
    print_n_bit_angles(num_c_qubits, c_st_list, ii + 1)

    c_st_list[ii] = 1
    print_n_bit_angles(num_c_qubits, c_st_list, ii + 1)


st_list = [None] * (num_c_qubits - 1)
print_n_bit_angles(num_c_qubits - 1, st_list, 0)

# %% [markdown]
# The empirical angles are very close to the ideal ones with shot noise.

# %% [markdown]
# Copyright &copy; 2026 D-Wave Systems, Inc
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
#
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This code example is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>
