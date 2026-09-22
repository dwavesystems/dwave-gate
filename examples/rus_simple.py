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

# %% [markdown]
# # Repeat Until Success with Error Checks
#
# Mid-circuit erasure detection (MCED) can be used to build up complex
# quantum circuits while occasionally *validating* that no error happened.
# This "repeat-until success" strategy utilizes advanced control flow during
# quantum algorithms, where the circuit restarts in unpredictable ways.
#
# The example below demonstrates this with a simple three-qubit circuit
# which, ideally, is just the identity. The circuit applies a three-qubit unitary
# $U$ followed by its inverse, back and forth, a given number of times `num_iterations`:
#
# $$
#     U^{-1} U \ldots U^{-1} U \; |\psi_0>
# $$
#
# After each $U$, an optional MCED is performed on all qubits. If any of
# the outcomes are bad, the circuit is restarted. For example, if
# `num_iterations=2`, the following timeline might arise which has one
# failed attempt:
#
# ```
# Reset qubits
# U
# mced -> no error detected
# U^-1
#
# U
# mced -> ERROR DETECTED
#
# Reset qubits
# U
# mced -> no error detected
# U^-1
#
# U
# mced -> no error detected
# U^-1
#
# measure qubits
# ```

# %% [markdown]
# ## Building the QCDL Program
#
# To simplify the QCDL program, abstract slightly by using procedures.
# `mced_check` is a procedure that performs MCEDs on all qubits and
# stores the value `1` into an `error_check_register`. The procedures
# `unitary` and `unitary_inverse` represent the operator $U$ (and its
# inverse) for the program.
#
# The control flow in this program is handled with a `Goto` expression.

# %%
from dwave.gate.qcdl import Scope, procedure, qcdl
from dwave.gate.qcdl.components import QCDLModule
from dwave.gate.qcdl.operations import cz, mced, measure, rx, ry, rz


@qcdl(num_qubits=3)
def main(repeat_until_success: bool, num_iterations: int = 4, **qubits):
    """Build up a unitary with optional repeat-until-success behavior.

    Args:
        repeat_until_success: When True, the circuit performs MCED checks
            throughout and restarts whenever an error is detected. Skipped
            when False.

        num_iterations: Number of (U^-1 MCED U) circuit components to
            perform. Circuit depth is controlled by `num_iterations`.

        **qubits: QCDL automatically passes qubits here with keys "q0", "q1", and "q2".
    """
    sc = Scope(*qubits.values())
    error_flag = sc.Register(name="error_flag")

    # Retry whenever errors are detected
    sc.Label("retry")
    for q in qubits.values():
        q.reset()

    for _ in range(num_iterations):
        # Perform the U operation.
        unitary(**qubits)

        # If RUS is on, check for errors and restart if any are detected.
        if repeat_until_success:
            mced_check(qubits, sc, error_flag_register=error_flag)
            with sc.If(None):
                # This condition is triggered when any error is detected
                sc.Goto("retry")

        # Undo the U operation
        unitary_inverse(**qubits)

    # Measure all qubits
    for q in qubits.values():
        measure(q)


@procedure
def unitary(q0, q1, q2):
    """Three-qubit unitary with no particular meaning."""
    rx(q0, 1.0)
    ry(q0, 0.5)
    ry(q1, -0.2)
    ry(q2, 4.0)
    rz(q2, 0.3)
    cz(q0, q1)
    cz(q0, q2)
    cz(q1, q2)


@procedure
def unitary_inverse(q0, q1, q2):
    """Hard-coded inverse of `unitary`."""
    cz(q1, q2)
    cz(q0, q2)
    cz(q0, q1)
    rz(q2, -0.3)
    ry(q2, -4.0)
    ry(q1, 0.2)
    ry(q0, -0.5)
    rx(q0, -1.0)


@procedure
def mced_check(
    qubits: dict[str, QCDLModule],
    sc: Scope,
    error_flag_register,
):
    """Perform an MCED on all qubits and check if any have errors."""
    for q in qubits.values():
        mced(q, register=error_flag_register)
    sc.all_to_all(error_flag_register == 1, reduce_op="|")


# %% [markdown]
# ## Execute without Error Checks and RTCF

# %%
from dwave.gate.leap import LeapQCDLSimulator

shots = 100
simulator = LeapQCDLSimulator()
future = simulator.run(
    main(repeat_until_success=False),
    shots=shots,
    noise_model=True,
    label="SDK Examples - Repeat Until Success without Error Checks",
)
results = future.result().result

results.get_counts(post_select=True)

# %% [markdown]
# ## Execute with Error Checks and RTCF

# %%
future = simulator.run(
    main(repeat_until_success=True),
    shots=shots,
    noise_model=True,
    label="SDK Examples - Repeat Until Success with Error Checks",
)
results = future.result().result

results.get_counts(post_select=True)

# %% [markdown]
# You might notice two things:
# - Why are fewer than `shots` outcomes returned, and
# - what is `post_select=True` for?
#
# The answer is that dual-rail qubits experience errors, and the simulator
# discards data with errors when `post_select=True`. To see this, try
# switching to `post_select=False` to observe all shots including data with
# errors labeled on specific qubits.
