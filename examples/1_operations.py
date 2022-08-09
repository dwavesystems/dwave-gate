# Confidential & Proprietary Information: D-Wave Systems Inc.
"""Using and appending operations."""
from dwgms.circuit import Circuit
from dwgms.operations.gates import Z, RY, CNOT, CX, Rotation

# Gates can be appended to the circuit (within a context) in several different
# ways, as detailed below.
circuit = Circuit(2)
with circuit.context as q:
    # append gates using keyword arg for qubits
    Z(qubits=q[0])
    # append parametric gates using positional args
    RY(4.2, q[1])
    # append parametric gates using qubit labels
    Rotation([0.1, 0.2, 0.3], "q0")
    # append multi-qubit gates using tuples
    CNOT((q[0], q[1]))
    # append multi-qubit gates using slicing
    CX(q[:2])

# Note that CNOT and CX are the exact same gate, and that CNOT is only an alias
# for CX (so they both are labelled as CX in the circuit).
print(circuit)

# We can also use the 'broadcast' method to apply operations to several qubits
# at once. The 'layered' method will append the operation on each qubit layering
# multi-qubit gates. The other method is called 'parallel' and will append the
# operation to each qubit without any overlaps.
circuit = Circuit(5)
with circuit.context as q:
    Z.broadcast(q, method="layered")

print(circuit)

# 'layered' is the default setting, but for single-qubit gates the outcome is
# the same as for 'parallel'.
circuit.reset()
with circuit.context as q:
    RY.broadcast(q, 0.3)

print(circuit)

# For multiqubit gates, the two methods differ.
circuit.reset()
with circuit.context as q:
    CNOT.broadcast(q, method="layered")

print(circuit)

circuit.reset()
with circuit.context as q:
    CNOT.broadcast(q, method="parallel")

print(circuit)

# Some gates implement decompositions as well.
print(Rotation.decomposition)
print(Rotation([0.1, 0.2, 0.3]).decomposition)

# Decompositions can be directly appended to circuits.
# NOTE: Doesn't work properly with parametric gates when called on an instance
# within a context (or any gate if called on an instance).
circuit.reset()
with circuit.context as q:
    Rotation([0.1, 0.2, 0.3], q[0]).decomposition

print(circuit)
