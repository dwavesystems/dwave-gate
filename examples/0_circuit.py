"""Building, modifying and excecuting a circuit."""
from dwgms.circuit import Circuit
from dwgms.operations.gates import X, RX, CNOT

# Create a circuit by defining the number of qubits (3) and the number of bits
circuit = Circuit(3, 2)

# Printing the circuit will print the circuits operations (currently empty).
print(repr(circuit))

# Printing the representation of the circuit, will provide some information
# (can be useful when using the Pyton interpreter or Jupyter).
print(repr(circuit))

# We can add more qubits to the circuit manually.
# Label is optional (otherwise one will be generated automatically)
circuit.add_qubit()
# Qubits in all registers can be accessed via 'circuit.gates'.
print(circuit.qubits)

# We can also, similarly, add a new quantum register and add a qubit to it.
circuit.add_qregister(label="reg")
circuit.add_qubit(qreg_label="reg")
print(circuit.qregisters)

# To create a circuit, we use the circuit context and append gates as follows.
with circuit.context as q:
    X(qubits=q[0])
    RX(0.5, qubits=q[2])

# We can also manually append a gate to the circuit, but first we need to unlock
# the circuit, since it's automatically locked when exiting a context.
circuit.unlock()
circuit.append(CNOT(qubits=("q0", "q1")))

print(circuit)
