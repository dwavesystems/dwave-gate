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
# ## Bayesian Decoding
# This notebook describes a method for recovering shots with
# erasures using Bayesian statistics and the tradeoff
# of fidelity incurred by doing so.

# This notebook requires Qiskit and Ocean software's Qiskit plugin.
# If you installed the Ocean SDK using the `pip install dwave-ocean-sdk[qiskit]`
# command, both are already installed.
# Otherwise, run the `pip install dwave-qiskit-plugin` command.

# %%
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# %%
num_qubits = 10
num_shots = 1000

# %% [markdown]
# ## Create a Circuit
# Create a random circuit where every ideal output has a fixed Hamming
# weight (i.e. a fixed number of 1's in the bitstring).

# %%
from bayesian_decoding.random_circuit_generation import (
    rand_circ_fixed_hamming_weight,
)
from qiskit import transpile

qc = rand_circ_fixed_hamming_weight(
    num_qubits=num_qubits,
    hamming_weight=num_qubits - 2,
    depth=50,
    seed=123,
    measure=False,
)
qc = transpile(qc, basis_gates=["cz", "rz", "sx"], optimization_level=2)
qc.count_ops()

# %% [markdown]
# ## Get the Exact Result

# %%
from qiskit.quantum_info import Statevector

sv = Statevector.from_instruction(qc)
print(sv.probabilities_dict(decimals=2))

# Measure all qubits
qc.measure_all()

# %% [markdown]
# ## Instantiate the D-Wave Qiskit Provider and Run the Circuit

# %%
from pprint import pp

from dwave.plugins.qiskit import DWaveProvider

with DWaveProvider() as provider:
    backend = provider.get_backend()
    job = backend.run(qc, shots=1000, noise_model=True)
    raw_counts = job.result().data()["raw_counts"]
    counts = job.result().get_counts()
pp(raw_counts)
pp(counts)

# %% [markdown]
# ## Check Yield from Performing Full Post Selection

# %%
job.result().data()["post_selection_yield"]

# %% [markdown]
# ## Optimize over the Beta Parameter in the Bayesian Decoding
# The beta parameter is essentially the
# strength of the "penalty" incurred
# by the difference in Hamming distance between two states.
# If matplotlib is installed, this creates a bar chart of the
# counts.
#

# %%
import numpy as np
from bayesian_decoding.bayesian_decoder import (
    DecoderParams,
    bayesian_decode_counts,
)
from qiskit.quantum_info import hellinger_fidelity

best_hf = 0.0
beta_values = np.linspace(1e-2, 5.0)
for beta in beta_values:
    decode_params = DecoderParams(
        alpha=0.0,
        beta=beta,
        max_distance=num_qubits // 2,
        max_erasures=num_qubits // 2,
        min_posterior=0.025,
        min_margin=0.025,
    )
    decoded_counts_tmp, assignments_tmp, discarded_tmp, posteriors_tmp = (
        bayesian_decode_counts(raw_counts, decoder_params=decode_params)
    )
    if hellinger_fidelity(decoded_counts_tmp, counts) > best_hf:
        decoded_counts = decoded_counts_tmp
        assignments = assignments_tmp
        discarded = discarded_tmp
        posteriors = posteriors_tmp
        best_hf = hellinger_fidelity(decoded_counts_tmp, counts)
        best_beta = beta

print(f"{best_beta=}")
print(hellinger_fidelity(decoded_counts, counts))
pp(decoded_counts)
pp(assignments)
pp(posteriors)

# %%
print(sum(counts.values()) / num_shots)
print(sum(decoded_counts.values()) / num_shots)

# %%
pp(hellinger_fidelity(decoded_counts, counts))
pp(hellinger_fidelity(decoded_counts, sv.probabilities_dict()))
pp(hellinger_fidelity(counts, sv.probabilities_dict()))

# %%
# from qiskit.visualization import plot_histogram

# plot_histogram(
#     [counts, decoded_counts],
#     legend=["Post Selected", "Bayesian Decoded"],
#     number_to_keep=10,
#     sort="value_desc",
# )
