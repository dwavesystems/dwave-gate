# Copyright 2022 D-Wave Systems Inc.
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

"""Creating and using templates."""
from dwave.gate.circuit import Circuit
from dwave.gate.tools import build_unitary
import dwave.gate.operations as ops

import numpy as np
np.set_printoptions(suppress=True)

# Let's create a circuit which applies a Z-gate to a single qubit, but by using
# only Hadamards and X-gates.
z_circuit = Circuit(1)
with z_circuit.context as (q, c):
    ops.Hadamard(q[0])
    ops.X(q[0])
    ops.Hadamard(q[0])

# We can check to see if it's correct by printing its unitary using the built-in 'build_unitary'
# function. The resulting matrix represents the unitary that is applied when applying these three
# operations according to the circuit above.
z_matrix = build_unitary(z_circuit)
print("Z matrix:\n", z_matrix)

# The above circuit can also be applied to other circuits. This will basically just apply the
# operations within that circuit to the new circuit, mapping the old qubits to the new.
circuit = Circuit(3)
with circuit.context as (q, c):
    z_circuit(q[1])

print("Circuit:", circuit.circuit)

# If the circuit is meant to be used as a custom operation, the 'create_operation' function can be
# used to transform the circuit in to an operation class, inheriting from the abstract 'Operations'
# class (as all other operations do).
from dwave.gate.operations import create_operation

# If no name is given, the custom operation class will simply be named "CustomOperation".
MyZOperation = create_operation(z_circuit, name="MyZOperation")

print(MyZOperation())

# We can compare its matrix representation to the built in 'Z' operation.
print("MyZOperation matrix:\n", MyZOperation.matrix)
print("Z matrix:\n", ops.Z.matrix)
assert np.allclose(MyZOperation.matrix, ops.Z.matrix)

# This custom gate can now be applied to the circuit in exactly the same way as any other gate. Note
# that if we don't reset the circuit, the 'z_circuit' operations will still be there.
circuit.reset()
with circuit.context as (q, c):
    MyZOperation(q[2])

print("Circuit:", circuit.circuit)
