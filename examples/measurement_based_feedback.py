# %% [markdown]
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
# # Simple Measurement-Based Feedback
#
# This notebook shows how to construct a measurement-based feedback circuit using QCDL
# and simulate it with a dual-rail qubit simulator. This example shows basic conditional logic
# in real time and sets the stage for more advanced examples.

# %%
import logging

# see the job IDs
logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Measurement-Based Feedback

# %%
from dwave.gate.qcdl import qcdl
from dwave.gate.qcdl.operations import h, measure, x


@qcdl(num_qubits=2)
def main(q0, q1):

    h(q1)  # Hadamard
    measure(q1)
    q0.sync(q1)  # Synchronize instruction execution of q0 and q1 in time

    with q0.If(condition=q1):
        x(q0)  # X gate (bit flip)

    measure(q0)


# %% [markdown]
# ## Simulate the circuit
# Initialize the QCDL simulator and run the QCDL

# %%
from dwave.gate.leap import LeapQCDLSimulator

shots = 100
simulator = LeapQCDLSimulator()
future = simulator.run(
    main(), shots=shots, noise_model=True, label="SDK Examples - Simple Measurement-Based Feedback"
)
results = future.result().result

# %% [markdown]
# Simulated measurement outcomes can be retrieved with `results.get_counts`

# %%
# Inspect measurements
counts = results.get_counts(register=["q0", "q1"], post_select=True)

print(counts)

# %% [markdown]
# You can see here that a bell-like state is created
# using real-time control flow. By conditioning the
# state of the second qubit on the measured state of
# the first, ideally this circuit outputs equal counts
# of |00> and |11>.
