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
from dwave.gate.circuit import Circuit, ParametricCircuit
from dwave.gate.tools import build_unitary
import dwave.gate.operations as ops

import numpy as np
np.set_printoptions(suppress=True)

# Let's create a parametric circuit which applies a rotation gate to a single qubit
rot_circuit = ParametricCircuit(1)
with rot_circuit.context as (p, q, c):
    ops.RZ(p[0], q[0])
    ops.RY(p[1], q[0])
    ops.RZ(p[2], q[0])

# The above circuit can be applied to other, non-parametric, circuits. This will just apply the
# operations within that circuit to the new circuit, mapping the old qubits to the new. Since it's a
# parametric circuit, the parameters need to be passed when calling it, similarly to how parametric
# operations work.
circuit = Circuit(3)
with circuit.context as (q, c):
    rot_circuit([np.pi, np.pi / 2, np.pi], q[1])

print("Circuit:", circuit.circuit)

# If the parametric circuit is meant to be used as a custom operation, the 'create_operation'
# function can be used to transform the circuit in to an parametric operation class, inheriting from
# the abstract 'ParametricOperations' class (as all other parametric operations do).
from dwave.gate.operations import create_operation

# If no name is given, the custom operation class will simply be named "CustomOperation".
MyRotOperation = create_operation(rot_circuit, name="MyRotOperation")

print(MyRotOperation([0, 1, 3.14]))

# We can compare its matrix representation to the built in 'Rotation' operation.
params = [1.2, 2.3, 3.4]
print("MyRotOperation matrix:\n", MyRotOperation(params).matrix)
print("Rotation matrix:\n", ops.Rotation(params).matrix)
assert np.allclose(MyRotOperation(params).matrix, ops.Rotation(params).matrix)

# This custom gate can now be applied to the circuit in exactly the same way as any other gate. Note
# that if we don't reset the circuit, the 'rot_circuit' operations will still be there.
circuit.reset()
with circuit.context as (q, c):
    MyRotOperation([3 * np.pi / 2, np.pi / 2, np.pi], q[2])

print("Circuit:", circuit.circuit)
