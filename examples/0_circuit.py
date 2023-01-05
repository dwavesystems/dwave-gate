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

"""Building, modifying and excecuting a circuit."""
from dwave.gate.circuit import Circuit
from dwave.gate.operations import X, RX, CNOT

# Create a circuit by defining the number of qubits (2)
circuit = Circuit(2)

# Printing the circuit will print the circuits operations (currently empty).
print("Circuit:", circuit.circuit)

# Printing the circuit object will provide some information which can be especially useful when
# using the Pyton interpreter or Jupyter.
print(circuit)

# We can add more qubits to the circuit manually.
circuit.add_qubit()

# Qubits in all registers can be accessed via 'circuit.qubits'.
print("Qubits:", circuit.qubits)

# We can also, similarly, add a new quantum register and add a qubit to it.
circuit.add_qregister(label="my_qreg")
circuit.add_qubit(qreg_label="my_qreg")
print("Quantum registers:", circuit.qregisters)

# Now, we should have 4 qubits in 2 registers.
print("Number of qubits:", circuit.num_qubits)
print("Qubits:", circuit.qubits)

# To create a circuit, we use the circuit context manager and append gates as follows;
# an X-gate to the first qubit (q[0]) and an X-rotation to the second qubit (q[1]).
with circuit.context as (q, c):
    X(q[0])
    RX(0.5, q[1])

print("Circuit", circuit.circuit)

# Note that 'circuit.context' returns a NamedTuple which is unpacked dirctly into 'c' and 'q' above.
# It would work equally well to simply keep the named tuple as e.g., 'reg' and then call the quantum
# and classical registers via 'reg.q' and 'reg.c' respectively:
#
#   with circuit.context as reg:
#       X(reg.q[0])
#       RX(0.5, reg.q[1])
#

# We now have a circuit object which we could, for example, send to run on a compatible
# simulator or hardware. After the circuit has been created it is automatically locked.
# This is to prevent any unexpected changes to the circuit. To unlock it, and add more
# gates to it, we simply call:
circuit.unlock()

# We can now apply more gates which will be appended to the circuit.
with circuit.context as (q, c):
    CNOT(q[0], q[1])

print("Circuit", circuit.circuit)

# If we later wish to reuse the circuit, we can reset it. This will clear all applied operations,
# but will keep the qubits and registers intact (unless 'keep_registers' is set to 'False').
circuit.reset(keep_registers=False)  # 'keep_registers=True' as default

print(circuit)
print("Circuit", circuit.circuit)
