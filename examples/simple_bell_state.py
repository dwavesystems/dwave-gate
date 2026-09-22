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
# # Simulating a Bell-State Measurement
#
# This notebook shows how to construct a Bell state using QCDL and
# simulate it with a dual-rail qubit simulator. Start here if you are new to
# QCDL and dual-rail qubit simulation.

# %%
import logging

# see the job IDs
logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Writing QCDL
#
# QCDL (Quantum Circuit Description Language) is a powerful way to specify
# quantum programs. To use it, use the decorator `@qcdl(num_qubits)`. For example,
#
# ```python
# @qcdl(num_qubits=1)
# def main(q):
#     q.h()
#     q.measure()
# ```
#
# says, "perform a Hadamard gate on one qubit and then measure it".
# The function `main` must have exactly one argument because the
# `qcdl` decorator is given `num_qubits=1`.

# %% [markdown]
# ## Bell Circuit in QCDL
#
#

# %%
from dwave.gate.qcdl import qcdl
from dwave.gate.qcdl.operations import cx, h, measure
from dwave.gate.utils.display import print_qcdl


# Remember to start with `@qcdl`. The `main` entry-point has two
# arguments `q0` and `q1` because you use `@qcdl(num_qubits=2)`.
@qcdl(num_qubits=2)
def main(q0, q1):

    h(q0)  # Hadamard
    cx(q0, q1)  # CNOT with q0 as control and q1 as target

    measure(q0)
    measure(q1)


# %% [markdown]
# ## Understanding QCDL Instructions.
#
# The function `main` corresponds to a quantum program specification.
# To see it, try calling `main` with `main()`.
# Alternatively, the `print_qcdl` function can be used to inspect a QCDL
# in a human-readable fashion.

# %%
print_qcdl(main())

# If your notebook doesn't display `print_qcdl` output well, try uncommenting the next line instead.
# print(print_qcdl(main(), to_Display=False))

# %% [markdown]
# ## Simulate and Retrieve Results

# %%
from dwave.gate.leap import LeapQCDLSimulator

shots = 100
simulator = LeapQCDLSimulator()
future = simulator.run(
    main(),
    shots=shots,
    noise_model=True,
    label="SDK Examples - Simulating a Bell-State Measurement",
)
results = future.result().result

# %% [markdown]
# Simulated measurement outcomes can be retrieved with `results.get_counts`

# %%
# Inspect measurements
counts = results.get_counts(register=["q0", "q1"], post_select=True)

print(counts)

# %% [markdown]
# You might notice two things:
# - Why are fewer than `shots` outcomes returned, and
# - what is `post_select=True` for?
#
# The answer is that dual-rail qubits experience errors, and the simulator
# discards data with errors when `post_select=True`. To see this, try
# switching to `post_select=False` to observe all shots including data with
# errors labeled on specific qubits.

# %%
