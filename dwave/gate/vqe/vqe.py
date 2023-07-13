# Copyright 2023 D-Wave Systems Inc.
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

__all__ = [
    "initial_parameters",
    "Measurements",
]

import warnings

import numpy as np

from dwave.gate.vqe.transformations.jordan_wigner import JordanWignerHamiltonian

try:
    from dwave.gate.vqe.cqm_partition import partition_cqm
except ImportError:
    cqm_supported = False
else:
    from dwave.cloud.exceptions import SolverFailureError

    cqm_supported = True


def initial_parameters(hamiltonian: JordanWignerHamiltonian):
    """TODO"""
    pauli_terms = hamiltonian.collect_pauli_terms()

    for string, coef in pauli_terms.items():
        if coef.imag != 0:
            # I have no idea what to do with the imaginary coefficients
            # For the trotterization we need only real coefficients
            # For now I take the norm, but this means something is either (1)
            # broken or (2) the ansatz differs from the literature in Anand et
            # al 2022 they use a slightly different ansatz but they say there
            # shouldn't be imag coefficients.....
            pauli_terms[string] = np.exp(coef).real
        else:
            pauli_terms[string] = np.exp(coef)

    # no idea if this is right
    return [val for _, val in pauli_terms.items()]


class Measurements:
    """TODO"""

    def __init__(self, hamiltonian: JordanWignerHamiltonian) -> None:
        self.pauli_terms = list(hamiltonian.collect_pauli_terms().keys())

        self.num_qubits = hamiltonian.num_orbitals
        self.num_measurements = len(self.pauli_terms)
        self.adjacency = None

    def generate_adjacency(self):
        """TODO"""
        self.adjacency = []
        for index, term in enumerate(self.pauli_terms):
            for previous_index, previous_term in enumerate(self.pauli_terms[:index]):
                commute_sum = self.num_qubits - len(term)  # commute on the identity
                for t in term.factors:
                    if (t.idx not in previous_term.indices) or previous_term[t.idx] == t:
                        commute_sum += 1
                if commute_sum % 2 == 0:
                    # two Pauli strings do not commute if and only if they do
                    # commute on an even number of indices
                    self.adjacency.append((index, previous_index))

    def _partition_linear(self):
        """TODO"""
        measurement_assignments = {}

        current_measurement = 0
        for term in range(self.num_measurements):
            new_measurement = any(
                [
                    ((i, term) in self.adjacency) or ((term, i) in self.adjacency)
                    for i in range(term)
                ]
            )
            if new_measurement:
                current_measurement += 1

            measurement_assignments[term] = current_measurement

        partition = {}

        for pauli_index, measurement_round in measurement_assignments.items():
            if measurement_round not in partition:
                partition[measurement_round] = [pauli_index]
            else:
                partition[measurement_round] = partition[measurement_round] + [pauli_index]

        return partition

    def partition(self, method: str = "best"):
        """TODO"""
        supported_methods = ["best", "cqm", "linear"]

        if method not in supported_methods:
            raise ValueError(f"method must be one of {supported_methods}, got {method}")

        if self.adjacency is None:
            self.generate_adjacency()

        if method == "cqm" or (method == "best" and cqm_supported):
            if not cqm_supported:
                raise ValueError("Cannot use CQM method unless dimod is installed.")

            try:
                partition = partition_cqm(self.adjacency, self.num_measurements)
            except SolverFailureError as e:
                # only proceed with 'linear' if 'method' is 'best'
                if method != "best":
                    raise e

                warnings.warn(str(e) + " Using 'linear' method instead.", stacklevel=2)
                method = "linear"
            else:
                return partition

        if method == "linear" or (method == "best" and not cqm_supported):
            partition = self._partition_linear()

        return partition
