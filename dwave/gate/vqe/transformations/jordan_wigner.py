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
    "JordanWignerAnnihilation",
    "JordanWignerCreation",
    "JordanWignerHamiltonian",
]

from itertools import product

import numpy as np

from dwave.gate.vqe.operators import Operator, OperatorTerm, Pauli


class JordanWignerAnnihilation(Operator):
    """TODO"""

    def __init__(self, index: int, length: int = None) -> None:
        values = []
        for _ in range(index):
            values.append(Pauli("Z"))

        values.append(
            OperatorTerm(
                "*",
                0.5,
                OperatorTerm(
                    "+", Pauli("X"), OperatorTerm("*", complex(0, 1), Pauli("Y"))
                ),
            )
        )

        for _ in range(index, length):
            values.append(1)

        super().__init__(values=values)


class JordanWignerCreation(Operator):
    """TODO"""

    def __init__(self, index: int, length: int = None) -> None:
        values = []
        for _ in range(index):
            values.append(Pauli("Z"))

        values.append(
            OperatorTerm(
                "*",
                0.5,
                OperatorTerm(
                    "+", Pauli("X"), OperatorTerm("*", complex(0, -1), Pauli("Y"))
                ),
            )
        )

        for _ in range(index, length):
            values.append(1)

        super().__init__(values=values)


class JordanWignerHamiltonian:
    """TODO"""

    def __init__(
        self, num_orbitals: int, first_order: np.ndarray, second_order: np.ndarray
    ) -> None:
        if (len(first_order.shape) != 2) or any([dim != num_orbitals for dim in first_order.shape]):
            raise ValueError(
                "First order interactions of wrong shape, expected shape "
                f"{(num_orbitals, num_orbitals)}, got {first_order.shape}"
            )

        if (len(second_order.shape) != 4) or any(
            [dim != num_orbitals for dim in second_order.shape]
        ):
            raise ValueError(
                "Second order interactions of wrong shape, expected shape "
                f"{(num_orbitals, num_orbitals, num_orbitals, num_orbitals)}, "
                f"got {second_order.shape}"
            )

        self.num_orbitals = num_orbitals

        self.terms = {}

        for idx_1, idx_2 in product(range(num_orbitals), repeat=2):
            a = JordanWignerAnnihilation(idx_1, num_orbitals)
            a_dagger = JordanWignerCreation(idx_2, num_orbitals)

            term = a_dagger * a

            term.reduce()

            self.terms[(idx_1, idx_2)] = (first_order[idx_1, idx_2], term.to_pauli_strings())

        for idx_1, idx_2, idx_3, idx_4 in product(range(num_orbitals), repeat=4):
            a_1 = JordanWignerAnnihilation(idx_1, num_orbitals)
            a_2 = JordanWignerAnnihilation(idx_2, num_orbitals)
            a_1_dagger = JordanWignerCreation(idx_3, num_orbitals)
            a_2_dagger = JordanWignerCreation(idx_4, num_orbitals)

            term = a_1_dagger * a_1 * a_2 * a_2_dagger

            term.reduce()

            self.terms[(idx_1, idx_2, idx_3, idx_4)] = (
                second_order[idx_1, idx_2, idx_3, idx_4],
                term.to_pauli_strings(),
            )

        self.pauli_terms = None

    def collect_pauli_terms(self):
        """TODO"""
        if self.pauli_terms is None:
            # assumes commutivity of sum terms (not sure if okay?)
            sum_of_coef = {}
            for item in self.terms.values():
                classical_coef = item[0]
                for qubit_coef, pauli_string in item[1]:

                    if pauli_string.is_identity:
                        continue

                    if pauli_string not in sum_of_coef:
                        sum_of_coef[pauli_string] = 0

                    sum_of_coef[pauli_string] += classical_coef * qubit_coef
            self.pauli_terms = sum_of_coef

        return self.pauli_terms

    def calculate_energy(self, measurements):
        """TODO"""
        energy = 0
        for item in self.terms.values():
            classical_coef = item[0]
            for qubit_coef, pauli_string in item[1]:
                if not pauli_string.is_identity and (pauli_string not in measurements):
                    raise ValueError(f"No measurement found for {pauli_string}")
                elif pauli_string.is_identity:
                    energy += classical_coef * qubit_coef
                else:
                    energy += classical_coef * qubit_coef * measurements[pauli_string]

        return energy

    def __repr__(self) -> str:
        return (
            f"Jordan-Wigner molecular hamiltonian with {self.num_orbitals} "
            f"orbitals and {len(self.terms)} hamiltonian terms"
        )

    def __str__(self) -> str:
        return self.__repr__()
