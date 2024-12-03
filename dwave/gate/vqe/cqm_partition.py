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
    "partition_cqm",
]

import warnings


try:
    import dimod

    from dwave.system import LeapHybridCQMSampler
except ImportError as e:
    raise ImportError("Dimod must be installed for CQM partioning.") from e


def partition_cqm(adjacency, num_measurements):
    """TODO"""
    cqm = dimod.ConstrainedQuadraticModel()

    measurement_rounds = [
        dimod.Integer(i, lower_bound=0, upper_bound=num_measurements)
        for i in range(num_measurements)
    ]

    cqm.set_objective(sum(measurement_rounds))

    for source, target in adjacency:
        cqm.add_constraint_from_comparison(
            measurement_rounds[source] ** 2
            + measurement_rounds[target] ** 2
            - 2 * measurement_rounds[source] * measurement_rounds[target]
            >= 1
        )

    sampler = LeapHybridCQMSampler()

    sampleset = sampler.sample_cqm(cqm)
    sampleset = sampleset.filter(lambda x: x.is_feasible)

    if len(sampleset) == 0:
        warnings.warn("No feasible measurement schemes found, defaulting to sequential measurement")
        cqm_sample = {i: i for i in range(num_measurements)}
    else:
        cqm_sample = sampleset.first.sample

    partition = {}

    for pauli_index, measurement_round in cqm_sample.items():
        if measurement_round not in partition:
            partition[measurement_round] = [pauli_index]
        else:
            partition[measurement_round] = partition[measurement_round] + [pauli_index]

    return partition
