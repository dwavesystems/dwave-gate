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
from __future__ import annotations

__all__ = [
    "tensor",
    "Tensor",
    "Pauli",
    "OperatorTerm",
    "Operator",
]

import copy
import numbers
from itertools import product
from typing import Any, List, Sequence, Tuple, Union

import numpy as np

def tensor(*factors: Sequence[Pauli]) -> Union[Tensor, Pauli]:
    """TODO"""
    if len(factors) == 0:
        return Pauli()
    if len(factors) == 1:
        return factors[0]

    return Tensor(*factors)


class Tensor:
    """TODO"""

    def __init__(self, *factors: Sequence[Pauli]) -> None:
        if len(factors) < 2:
            raise ValueError("Minimum of 2 Pauli objects required to create a tensor.")
        for f in factors:
            if not isinstance(f, Pauli):
                raise TypeError("Only Pauli object are accepted factors.")

        # store as tuple to make immutable/hashable
        self._factors = tuple(factors)

    @property
    def indices(self) -> List[int]:
        """TODO"""
        return [f.idx for f in self.factors]

    @property
    def factors(self) -> Tuple[Pauli, ...]:
        """TODO"""
        return self._factors

    @property
    def is_identity(self) -> bool:
        """TODO"""
        return all(f.is_identity for f in self.factors)

    def __matmul___(self, pauli: Tensor) -> Tensor:
        return Tensor(*self.factors, pauli)

    def __len__(self) -> int:
        return len(self.factors)

    def __getitem__(self, item: int) -> Pauli:
        return self.factors[item]

    def __reversed__(self) -> Union[Tensor, Pauli]:
        if isinstance(self, Pauli):
            return self
        return Tensor(*self.factors[::-1])

    def __eq__(self, tensor: Tensor) -> bool:
        if isinstance(tensor, Tensor) and self.factors == tensor.factors:
            return True
        return False

    def __hash__(self) -> int:
        return hash(self.factors)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return " @ ".join(map(str, self.factors))


class Pauli(Tensor):
    """TODO"""

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, complex(0, 1)], [-complex(0, 1), 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Id = np.array([[1, 0], [0, 1]], dtype=complex)

    pauli_algebra = {"X": X, "Y": Y, "Z": Z, "I": Id}

    def __init__(self, name=None, idx=None) -> None:
        if name and name not in Pauli.pauli_algebra.keys():
            raise ValueError(
                f"Name not in pauli algebra, must be one of {Pauli.pauli_algebra.keys()}"
            )

        self.name = name or "I"
        self.idx = idx
        self.value = Pauli.pauli_algebra[self.name]
        self._factors = [self]

    @property
    def is_identity(self) -> bool:
        """TODO"""
        return self.name == "I"

    @staticmethod
    def match_pauli_matrix(p: Pauli) -> Tuple[str, complex, List]:
        """TODO"""
        if np.equal(np.identity(2, dtype=complex), p).all():
            return 1

        matches = [key for key, val in Pauli.pauli_algebra.items() if np.equal(val, p).all()]

        if len(matches) == 1:
            return matches[0]

        matches = [
            key
            for key, val in Pauli.pauli_algebra.items()
            if np.equal(val, p * complex(0, 1)).all()
        ]
        if len(matches) == 1:
            return ("*", complex(0, 1), matches[0])

        matches = [
            key for key, val in Pauli.pauli_algebra.items() if np.equal(val, -1 * p).all()
        ]
        if len(matches) == 1:
            return ("*", -1, matches[0])

        matches = [
            key
            for key, val in Pauli.pauli_algebra.items()
            if np.equal(val, p * complex(0, -1)).all()
        ]
        if len(matches) == 1:
            return ("*", complex(0, -1), matches[0])


    def __matmul___(self, pauli: Pauli) -> Tensor:
        return Tensor(self, pauli)

    def __mul__(self, other: Any) -> Tuple[str, Pauli, Any]:
        if isinstance(other, Pauli):
            return Pauli.match_pauli_matrix(self.value @ other.value)
        else:
            return ("*", self, other)

    def __eq__(self, pauli: Pauli) -> bool:
        if isinstance(pauli, Pauli) and self.name == pauli.name and self.idx == pauli.idx:
            return True
        return False

    def __hash__(self) -> int:
        return hash(self.name) + hash(self.idx)

    def __str__(self) -> str:
        if self.idx is not None:
            return f"{self.name}{self.idx}"
        return f"{self.name}"

    def __repr__(self) -> str:
        return str(self)


class OperatorTerm:
    """TODO"""

    supported_operators = ["*", "+"]

    def __init__(self, operator: str, left, right) -> None:
        if operator not in OperatorTerm.supported_operators:
            raise ValueError(
                f"Operator not supported, must be one of {OperatorTerm.supported_operators}"
            )

        self.operator = operator
        self.left = left
        self.right = right

    def reduce(self):
        """TODO"""
        if isinstance(self.left, OperatorTerm):
            self.left = self.left.reduce()

        if isinstance(self.right, OperatorTerm):
            self.right = self.right.reduce()

        # distributive law
        if self.operator == "*":
            # a*(x + y) => a*x + a*y
            if isinstance(self.right, OperatorTerm) and self.right.operator == "+":
                self.operator = "+"
                child_left = self.left
                self.left = OperatorTerm("*", child_left, self.right.left)
                self.right = OperatorTerm("*", child_left, self.right.right)
            # (a + b)*x => a*x + b*x
            elif isinstance(self.left, OperatorTerm) and self.left.operator == "+":
                self.operator = "+"
                child_right = self.right
                # must be done in this order to not mess up the state
                self.right = OperatorTerm("*", self.left.right, child_right)
                self.left = OperatorTerm("*", self.left.left, child_right)

        # simplification of the same type
        if isinstance(self.left, numbers.Number) and isinstance(self.right, numbers.Number):
            if self.operator == "*":
                return self.left * self.right
            if self.operator == "+":
                return self.left + self.right
        elif isinstance(self.left, np.ndarray) and isinstance(self.right, np.ndarray):
            if self.operator == "*":
                return self.left @ self.right
            if self.operator == "+":
                return self.left + self.right
        elif isinstance(self.left, Pauli) and isinstance(self.right, Pauli):
            if self.operator == "*":
                return_value = self.left * self.right
                if isinstance(return_value, numbers.Number):
                    return return_value
                else:
                    return OperatorTerm(*return_value)

        # identity simplification
        if self.operator == "*":
            if isinstance(self.left, numbers.Number) and self.left == 0:
                return 0
            elif isinstance(self.right, numbers.Number) and self.right == 0:
                return 0
            if isinstance(self.left, numbers.Number) and self.left == 1:
                return self.right
            elif isinstance(self.right, numbers.Number) and self.right == 1:
                return self.left
        elif self.operator == "+":
            if isinstance(self.left, numbers.Number) and self.left == 0:
                return self.right
            elif isinstance(self.right, numbers.Number) and self.right == 0:
                return self.left

        return self

    def split(self):
        """TODO"""
        split_terms = []

        if self.operator == "+":
            if isinstance(self.left, OperatorTerm):
                split_terms.extend(self.left.split())
            else:
                split_terms.append(self.left)

            if isinstance(self.right, OperatorTerm):
                split_terms.extend(self.right.split())
            else:
                split_terms.append(self.right)

        else:
            split_terms = [self]

        return split_terms

    def iter_left(self, order="bottom->top"):
        """TODO"""
        orders = ["bottom->top", "top->bottom"]

        if order not in orders:
            raise ValueError(f"method must be one of {orders}, got {order}")

        if order == "top->bottom":
            yield self.left

        if isinstance(self.left, OperatorTerm):
            yield from self.left.iter_left()

        if order == "bottom->top":
            yield self.left

    def iter_right(self, order="bottom->top"):
        """TODO"""
        orders = ["bottom->top", "top->bottom"]

        if order not in orders:
            raise ValueError(f"method must be one of {orders}, got {order}")

        if order == "top->bottom":
            yield self.right

        if isinstance(self.left, OperatorTerm):
            yield from self.right.iter_right()

        if order == "bottom->top":
            yield self.right

    def traverse(self, order="left->right"):
        """TODO"""
        orders = ["left->right", "right->left"]

        if order not in orders:
            raise ValueError(f"method must be one of {orders}, got {order}")

        yield self

        if order == "left->right":
            if isinstance(self.left, OperatorTerm):
                yield from self.left.traverse(order)

            if isinstance(self.right, OperatorTerm):
                yield from self.right.traverse(order)

        elif order == "right->left":
            if isinstance(self.right, OperatorTerm):
                yield from self.right.traverse(order)

            if isinstance(self.left, OperatorTerm):
                yield from self.left.traverse(order)

    def iter_leaves(self, order="left->right"):
        """TODO"""
        orders = ["left->right", "right->left"]

        if order not in orders:
            raise ValueError(f"method must be one of {orders}, got {order}")

        if order == "left->right":
            if isinstance(self.left, OperatorTerm):
                yield from self.left.iter_leaves(order)
            else:
                yield self.left

            if isinstance(self.right, OperatorTerm):
                yield from self.right.iter_leaves(order)
            else:
                yield self.right

        elif order == "right->left":
            if isinstance(self.right, OperatorTerm):
                yield from self.right.iter_leaves(order)
            else:
                yield self.right

            if isinstance(self.left, OperatorTerm):
                yield from self.left.iter_leaves(order)
            else:
                yield self.left

    def homogenize(self, index=0):
        """TODO"""
        copy_self = copy.copy(self)

        substituion_dict = {}
        substituion_dict_left = {}
        substituion_dict_right = {}

        if isinstance(self.left, OperatorTerm):
            if self.left.operator == self.operator:
                copy_self.left, substituion_dict_left = self.left.homogenize(
                    index + 1
                )  # indexes arent unique, a bug!
            else:
                substituion_dict[f"X_{index}"] = self.left
                copy_self.left = f"X_{index}"
                index += 1

        if isinstance(self.right, OperatorTerm):
            if self.right.operator == self.operator:
                copy_self.right, substituion_dict_right = self.right.homogenize(index + 1)
            else:
                substituion_dict[f"X_{index}"] = self.right
                copy_self.right = f"X_{index}"
                index += 1

        # NOTE: Only Python >=3.9
        # return copy_self, substituion_dict | substituion_dict_left | substituion_dict_right
        return copy_self, {**substituion_dict, **substituion_dict_left, **substituion_dict_right}

    def collect(self):
        """TODO"""
        if self.operator == "+":
            unit = 0
        elif self.operator == "*":
            unit = 1

        old_term_homogenized, substitutions = self.homogenize()

        final_terms = {}
        for term in old_term_homogenized.iter_leaves():
            if isinstance(term, numbers.Number):
                if "scalar" not in final_terms:
                    final_terms["scalar"] = unit

                if self.operator == "+":
                    final_terms["scalar"] = final_terms["scalar"] + term
                elif self.operator == "*":
                    final_terms["scalar"] = final_terms["scalar"] * term

            else:
                if str(term.__class__) not in final_terms:
                    if term in substitutions:
                        term = substitutions[term].collect()
                    final_terms[str(term.__class__)] = term
                elif isinstance(term, np.ndarray):
                    # there is a slight bug here
                    # I'm assuming that the numpy matrix commutes with the Pauli matracies
                    # which in general isn't true
                    if self.operator == "+":
                        final_terms[str(term.__class__)] = final_terms[str(term.__class__)] + term
                    elif self.operator == "*":
                        final_terms[str(term.__class__)] = final_terms[str(term.__class__)] * term
                elif isinstance(term, Pauli):
                    if self.operator == "+":
                        final_terms[str(term.__class__)] = OperatorTerm(
                            "+", final_terms[str(term.__class__)], term
                        )
                    elif self.operator == "*":
                        prod = final_terms[str(term.__class__)] * term
                        if prod == 1:
                            del final_terms[str(term.__class__)]
                        elif len(prod) > 1 and prod[0] == "*":
                            # this section has some assumptions about the output type of PauliMatrix multiplication
                            if isinstance(prod[1], numbers.Number):
                                final_terms["scalar"] = final_terms["scalar"] * prod[1]
                            elif isinstance(prod[1], Pauli):
                                final_terms[str(term.__class__)] = prod[1]

                            if isinstance(prod[2], numbers.Number):
                                final_terms["scalar"] = final_terms["scalar"] * prod[2]
                            elif isinstance(prod[2], Pauli):
                                final_terms[str(term.__class__)] = prod[2]
                else:
                    if term in substitutions:
                        term = substitutions[term].collect()
                    final_terms[str(term.__class__)] = OperatorTerm(
                        self.operator, final_terms[str(term.__class__)], term
                    )

        final_item = unit

        for _, item in final_terms.items():
            final_item = OperatorTerm(self.operator, final_item, item)

        return final_item

    def __mul__(self, other):
        return OperatorTerm("*", self, other)

    def __add__(self, other):
        return OperatorTerm("+", self, other)

    def __str__(self) -> str:
        return "(" + str(self.operator).join([str(self.left), str(self.right)]) + ")"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other):
        # super strict version of equality - for example non-associative
        # subsuming some errors in favor of easy typing
        try:
            return (self.left == other.left) and (self.right == other.right)
        except Exception as e:
            return False


class Operator:
    """TODO"""

    def __init__(self, length: int = None, values=None, default_value=0) -> None:
        if (values is None) and (length is None):
            raise ValueError(f"Either length of values must be specified")

        if values is None:
            self.operator_terms = [default_value for _ in range(length)]
        elif (length is None) or (len(values) == length):
            self.operator_terms = values
        else:
            raise ValueError(f"Wrong length of values: given {len(values)}, expected {length}")

        self.length = len(self.operator_terms)

    def reduce(self):
        """TODO"""
        for index, term in enumerate(self.operator_terms):
            if isinstance(term, OperatorTerm):
                reduced = False
                while not reduced:
                    old_term = copy.copy(self.operator_terms[index])
                    self.operator_terms[index] = self.operator_terms[index].reduce()
                    if (
                        not isinstance(self.operator_terms[index], OperatorTerm)
                        or old_term == self.operator_terms[index]
                    ):
                        reduced = True
                    if isinstance(self.operator_terms[index], OperatorTerm):
                        self.operator_terms[index] = self.operator_terms[index].collect()

                    if isinstance(self.operator_terms[index], OperatorTerm):
                        self.operator_terms[index] = self.operator_terms[index].reduce()

                    if not reduced and not isinstance(self.operator_terms[index], OperatorTerm):
                        reduced = True

    def split(self):
        """TODO"""
        return_values = []
        for index in range(self.length):
            if isinstance(self.operator_terms[index], OperatorTerm):
                return_values.append(self.operator_terms[index].split())
            else:
                return_values.append([self.operator_terms[index]])

        return [Operator(values=i) for i in product(*return_values)]

    def collect(self):
        """TODO"""
        for index in range(self.length):
            if isinstance(self.operator_terms[index], OperatorTerm):
                self.operator_terms[index] = self.operator_terms[index].collect()

    def to_pauli_strings(self):
        """TODO"""
        split_operators = self.split()

        pauli_strings = []
        for operator in split_operators:
            coefficient = 1
            paulis = []
            for index, term in enumerate(operator.operator_terms):
                if isinstance(term, numbers.Number):
                    coefficient *= term
                elif isinstance(term, Pauli):
                    paulis.append(Pauli(term.name, index))
                elif isinstance(term, OperatorTerm):
                    if not (
                        isinstance(term.left, numbers.Number) or isinstance(term.left, Pauli)
                    ):
                        raise ValueError(
                            f"term left child is of unsupported class {term.left.__class__}, likely not simplified enough"
                        )

                    if not (
                        isinstance(term.right, numbers.Number)
                        or isinstance(term.right, Pauli)
                    ):
                        raise ValueError(
                            f"term right child is of unsupported class {term.left.__class__}, likely not simplified enough"
                        )

                    for child in [term.left, term.right]:
                        if isinstance(child, numbers.Number):
                            coefficient *= child
                        elif isinstance(child, Pauli):
                            paulis.append(Pauli(child.name, index))
                else:
                    raise ValueError(
                        f"term {term} of class {term.__class__} not a Number, PauliMatrix, or OperatorTerm"
                    )
            pauli_strings.append((coefficient, tensor(*paulis)))
        return pauli_strings

    def __mul__(self, other):
        if not isinstance(other, Operator):
            raise ValueError(
                f"Operators can only be multiplied with operators: got {other.__class__}"
            )

        if self.length != other.length:
            raise ValueError(
                f"Operators must be of same length: got {self.length} and {other.length}"
            )

        return Operator(
            values=[
                OperatorTerm("*", i, j) for i, j in zip(self.operator_terms, other.operator_terms)
            ]
        )

    def _add__(self, other):
        if not isinstance(other, Operator):
            raise ValueError(
                f"Operators can only be multiplied with operators: got {other.__class__}"
            )

        if self.length != other.length:
            raise ValueError(
                f"Operators must be of same length: got {self.length} and {other.length}"
            )

        return Operator(
            values=[
                OperatorTerm("+", i, j) for i, j in zip(self.operator_terms, other.operator_terms)
            ]
        )

    def __str__(self) -> str:
        return "$" + " \otimes ".join([str(i) for i in self.operator_terms]) + "$"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Operator):
            raise ValueError(
                f"Operators can only be multiplied with operators: got {other.__class__}"
            )

        return all([i == j for i, j in zip(self.operator_terms, other.operator_terms)])
