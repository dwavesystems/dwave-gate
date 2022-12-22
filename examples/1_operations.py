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

"""Using and appending operations."""
from dwave.gate.circuit import Circuit
import dwave.gate.operations as ops

# Operations, or gates, are objects that contain information related to that specific operation,
# including its matrix representation, potential decompositions, and application within a circuit.
print(ops.X())

# We can call the matrix property on either the class itself or on an instance (which is an
# instantiated operation, with additional parameters and/or qubit information).
print("X matrix:\n", ops.X.matrix)

# If the matrix representation isn't general (e.g., as for the X-gate above) and requires knowledge
# of parameters, it can only be retrieved from an instance (otherwise an exception will be raised).
print("Z-rotation matrix:\n", ops.RZ(4.2).matrix)

# Operations are applied when either the class or an instance is called within a circuit context.
# They can be applied to the circuit in several different ways, as detailed below. Both qubits and
# parameters can be passed either as single values (if supported by the gate) or contained in a
# sequence.
circuit = Circuit(3)
with circuit.context as (q, c):
    # apply single-qubit gate
    ops.Z(q[0])
    # apply single-qubit gate using kwarg
    ops.Hadamard(qubits=q[0])
    # apply a parametric gate
    ops.RY(4.2, q[1])
    # apply a parametric gate using kwargs
    ops.Rotation(parameters=[4.2, 3.1, 2.3], qubits=q[1])
    # apply controlled gate
    ops.CNOT(q[0], q[1])
    # apply controlled gate using kwargs
    ops.CX(control=q[0], target=q[1])
    # apply controlled qubit gates using slicing (must unpack)
    ops.CZ(*q[:2])
    # apply multi-qubit (non-controlled) gates (note tuple)
    ops.SWAP((q[0], q[1]))
    # apply multi-qubit (non-controlled) gates using kwargs
    ops.CSWAP(qubits=(q[0], q[1], q[2]))
    # apply multi-qubit (non-controlled) gates using slicing
    ops.SWAP(q[:2])
    # apply gate on all qubits in the circuit
    ops.Toffoli(q)

print(circuit)

# Print all operations in the circuit ()
for op in circuit.circuit:
    print(op)

# Note that e.g., CNOT and CX apply the exact same gate. CNOT is only an alias for CX (so they both
# are labelled as CX in the circuit). There are also other aliases which you can spot either in the
# `dwave/gate/operations/operations.py` file, or read more about in the documentation.
