# %% [markdown]
# Copyright &copy; 2026 D-Wave
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

# %% [markdown]
# # Real-Time Feedback Using Boolean Expressions
#
# This notebook shows how to construct an advanced measurement-based
# feedback circuit, which uses Boolean expressions in QCDL, and simulate it
# with a dual-rail qubit simulator. This example introduces you to the basics
# of real-time calculations and sets the stage for more advanced arithmetic
# examples.

# %%
import logging

# see the job IDs
logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Conditional Branching & Registers

# %%
from dwave.gate.qcdl import Scope, qcdl
from dwave.gate.qcdl.operations import measure, x


@qcdl(num_qubits=2)
def main(q0, q1):

    sc = Scope(q0, q1)  # Convenience method to utilize qubits as a group

    c0 = sc.Register(
        0, name="creg0"
    )  # Instantiate 18-bit classical register c0 to a value 0
    c1 = sc.Register(0, name="creg1")  # Register for the 2nd qubit

    measure(q0, register=c0)  # Measure and store result in c0
    q0.sync(q1)  # Synchronize instruction execution of q0 and q1 in time

    # QCDL conditional instruction If(condition = c0 or c1), performing a logical Boolean OR in real-time
    with q1.If(c0 | c1 == 1):
        x(q1)  # X gate (bit flip)

    measure(q1)


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
    label="Real-Time Feedback Using Boolean Expressions")
results = future.result().result

# %% [markdown]
# Simulated measurement outcomes can be retrieved with `results.get_counts`

# %%
# Inspect measurements
counts = results.get_counts(register=["q0", "q1"], post_select=True)

print(counts)
