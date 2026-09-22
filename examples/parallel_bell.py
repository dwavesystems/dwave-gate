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
# # Simulating Parallel Circuits
#
# For small circuits such as a two-qubit Bell circuit, there is an opportunity to
# boost throughput by running multiple copies of the same circuit in parallel.
#
# This notebook shows how to perform this operation and simulate it. The technique
# used here is ''erasure-aware'': when multiple circuits are executed in parallel,
# one of the following possibilities can happen:
# 1. Both circuits experience an erasure
# 2. One circuit experiences an erasure
# 3. Neither circuit experiences an erasure
#
# The technique in this notebook handles these cases optimally: in the first case,
# the shot is discarded. In the second case, one data point is returned. In the
# third case, two data points are returned. In this way, data from the two circuits
# are used whenever possible.

# %%
import logging
from collections import defaultdict

from dwave.gate.qcdl import procedure, qcdl
from dwave.gate.qcdl.operations import cx, h, measure
from dwave.gate.utils.display import print_qcdl

logging.basicConfig(level=logging.INFO)


# %%
def post_select_and_aggregate(raw_counts: dict[str, int]) -> defaultdict[int]:
    """Aggregate raw counts of 2 subcircuits to 2-qubit counts
    while excluding leakage in subcircuits.

    Args:
        raw_counts : Counts with all information, including erasure.

    Returns:
        Count dict of counts where one of the parallel bell states
        had no erasures

    """
    aggregated_counts = defaultdict(int)
    for bstr, count in raw_counts.items():
        c1_result = bstr[:2]
        c2_result = bstr[2:4]
        if "*" not in c1_result:
            aggregated_counts[c1_result] += count
        if "*" not in c2_result:
            aggregated_counts[c2_result] += count
    print(aggregated_counts)
    shot_yield = sum(aggregated_counts.values()) / sum(raw_counts.values())
    print("The yield is", shot_yield)
    return aggregated_counts


# %% [markdown]
# ## Parallel Bell Circuits in QCDL
# Use 4 qubits to run two GHz circuits. For convenience, first define a `Bell_circuit`
# function for a sub-circuit. The use of the `procedure` decorator is optional.
#


# %%
@procedure
def Bell_circuit(q0, q1):
    h(q0)
    cx(q0, q1)


# %%
@qcdl(4)
def main(**kwargs):

    q = dict(enumerate(kwargs.values()))

    Bell_circuit(q[0], q[1])
    Bell_circuit(q[2], q[3])

    for qubit in q.values():
        measure(qubit)


# %% [markdown]
# Print the QCDL instructions

# %%
print_qcdl(main())

# If your notebook doesn't display `print_qcdl` output well, try uncommenting the next line instead.
# print(print_qcdl(main(), to_Display=False))

# %% [markdown]
# ## Simulate and Retrieve Results
#
#

# %%
from dwave.gate.leap import LeapQCDLSimulator

shots = 100
simulator = LeapQCDLSimulator()
future = simulator.run(
    main(),
    shots=shots,
    noise_model=True,
    label="SDK Examples - Simulating Parallel Circuits",
)
results = future.result().result

# %% [markdown]
# Simulated measurement outcomes can be retrieved with `results.get_counts`.
# Print the results without post-selection to show erasure errors.

# %%
# Inspect measurements
raw_counts = results.get_counts(
    post_select=False,
)[0]


print(raw_counts)

# %% [markdown]
# Post process the results to get the counts for 2-qubit groups. When
# erasure only happens in one of the sub-circuits, the result of the other
# sub-circuit is still valid (i.e. if you get a result of `*011`, the `11` result is
# still counted as 'no error'). Also compute the yield (fraction of shots with no erasures)
# of this method: it is roughly double the simple version of Bell circuit.

# %%
aggregated_counts = post_select_and_aggregate(raw_counts)

# %% [markdown]
# This notebook demonstrates that by running parallel sub-circuits on one QPU,
# yield can be greatly boosted and exceed 1.0. That is very useful in near-term, if
# one is able to run multiple instances of the same Bell circuit (or another circuit of
# interest) across different pairs of qubits.
