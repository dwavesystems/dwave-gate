# %% [markdown]
# ## Bayesian Decoding

# %%
nq = 10
num_shots = 1000

# %% [markdown]
# ## Create a Circuit
# This creates a random circuit where every ideal output has a fixed hamming weight (i.e. a fixed number of 1's in the bitstring)

# %%
from qiskit import transpile

from examples.error_detection_handling.utils.random_circuit_generation import (
    rand_circ_fixed_hamming_weight,
)

qc = rand_circ_fixed_hamming_weight(
    num_qubits=nq, hamming_weight=nq - 2, depth=50, seed=123, measure=False
)
qc = transpile(qc, basis_gates=["cz", "rz", "sx"], optimization_level=2)
qc.count_ops()

# %% [markdown]
# ## Get the exact result

# %%
from qiskit.quantum_info import Statevector

sv = Statevector.from_instruction(qc)
print(sv.probabilities_dict(decimals=2))

# Measure all qubits
qc.measure_all()

# %% [markdown]
# ## Instantiate the DWave qiskit provider and run the circuit

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
# ## Check the yield from performing full post selection

# %%
job.result().data()["post_selection_yield"]

# %% [markdown]
# ## Run the circuit on the Leap simulator backend

# %% [markdown]
# ## Optimize over the Beta parameter in the Bayesian Decoding

# %%
import numpy as np
from qiskit.quantum_info import hellinger_fidelity

from examples.error_detection_handling.bayesian_decoding.bayesian_decoder import (
    DecoderParams,
    bayesian_decode_counts,
)

best_hf = 0.0
beta_values = np.linspace(1e-2, 5.0)
for beta in beta_values:
    decode_params = DecoderParams(
        alpha=0.0,
        beta=beta,
        max_distance=nq // 2,
        max_erasures=nq // 2,
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
from qiskit.visualization import plot_histogram

plot_histogram(
    [counts, decoded_counts],
    legend=["Post Selected", "Bayesian Decoded"],
    number_to_keep=10,
    sort="value_desc",
)
