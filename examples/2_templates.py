# Confidential & Proprietary Information: D-Wave Systems Inc.
"""Creating and using templates."""
from dwgms.circuit import Circuit
from dwgms.operations import X, RY, RZ, Hadamard
from dwgms.operations.gates import Rotation
from dwgms.operations.template import template

import numpy as np

# Let's create a circuit which applies a Z-gate to a single qubit, but by using
# only Hadamards and X-gates.
z_gate_circuit = Circuit(1)
with z_gate_circuit.context as q:
    Hadamard(q[0])
    X(q[0])
    Hadamard(q[0])

# We can see that it's correct by printing it's unitary.
print(z_gate_circuit.build_unitary())

# The above circuit can also be appended to other circuits.
circuit = Circuit(5)
with circuit.context as q:
    z_gate_circuit(q[1])

print(circuit)


# A better way to create a custom gate is to use the template decorator. We can
# write a class with a 'circuit' method, which applies the circuit, and two
# class attributes 'num_qubits' and 'num_params'. Below is a custom rotation
# gate implementation using only 'RZ' and 'RY' gates.
@template
class RotGate:
    num_qubits = 2
    num_params = 3

    def circuit(self):
        RZ(self.params[0], self.qubits[0])
        RY(self.params[1], self.qubits[1])
        RZ(self.params[2], self.qubits[0])


# We can print the new gate representation.
rotation_gate = RotGate([0.1, 0.2, 0.3])
print(rotation_gate)

# And compare it's matrix to the built in 'Rotation' gate.
print(rotation_gate.matrix)
print(Rotation([0.1, 0.2, 0.3]).matrix)
assert np.allclose(rotation_gate.matrix, Rotation([0.1, 0.2, 0.3]).matrix)

# This custom gate can now be applied to the circuit in exactly the same way as
# any other gate. Note that the 'z_gate_circuit' is still there.
circuit.unlock()
with circuit.context as q:
    RotGate([1, 2, 3], (q[1], q[3]))

print(circuit)
