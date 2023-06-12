# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: dwave-gate
#     language: python
#     name: python3
# ---

"""
# Beginner's Guide to `dwave-gate`

`dwave-gate` lets you easily construct and simulate quantum circuits.

This tutorial guides you through using the `dwave-gate` library to inspect,
construct circuits from, and simulate quantum gates using a performant
state-vector simulator.

## Circuits and Operations
Begin by importing the necessary modules.

*   `Circuit` objects contain the full quantum circuit, with all the operations,
    measurements, qubits and bits.
*   Operations, or gates, are objects that contain information related to a
    particular operation, including its matrix representation, potential
    decompositions, and how to apply it to a circuit.
"""

from dwave.gate.circuit import Circuit
import dwave.gate.operations as ops

###############################################################################
# You can use the `operations` module (shortened above to `ops`) to access a
# variety of quantum gates; for example, a Pauli X operator.

ops.X()

###############################################################################
# Notice above that an operation can be instantiated without declaring which qubits
# it should be applied to.
#
# The matrix property can be accessed either via the gate class itself or an
# instance (i.e., an instantiated operation, which can contain additional
# parameters and qubit information).

ops.X.matrix

###############################################################################
# If the matrix representation is dependent on parameters (e.g., the X-rotation operation) it can
# only be retrieved from an instance.


ops.RZ(4.2).matrix

###############################################################################
# ## Circuit Context
# Operations are applied by calling, within the context of a circuit, either a
# class or an instance of a class.
#
# You can apply operations to a circuit in several different ways, as demonstrated
# below. You can pass both qubits and parameters as either single values (when
# supported by the gate) or sequences. Note that different types of gates accept
# slightly different arguments, although you can _always_ pass the qubits as
# sequences via the keyword argument `qubits`.
#
# Always apply any operations you instantiate within a circuit context to
# specific qubits in the circuit's qubit register. You can access the qubit
# register via the named tuple returned by the context manager as `q`, indexing
# into it to retrieve the corresponding qubit.

###############################################################################
# ## Registers
# This example starts by creating a circuit object with two qubits in its
# register and applying a single X gate to the first qubit.

circuit = Circuit(2)

with circuit.context as reg:
    ops.X(reg.q[0])

###############################################################################
# You can access the qubit register via the `Circuit.qregisters` attribute.
#
# In the current example, this attribute contains a single qubit register with
# two qubits; it could contain any number of qubit registers. You can
#
# * add another register with the `Circuit.add_qregister(n)` method, where `n`
#   is the number of qubits in the new register
# * add a qubit with `Circuit.add_qubit()`, optionally passing a qubit object
#   and/or a register to which to add it.
#
# The registers tuple can also be unwrapped directly into a qubit register `q` (and a
# classical register `c`).


###############################################################################
# ## Example: Applying Various gates to a Circuit


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

###############################################################################
# Print the circuit above to see general information about it: type of circuit,
# number of qubits/bits, and number of operations.

print(circuit)

###############################################################################
# You can also access operations in a circuit using the `Circuit.circuit`
# attribute. The code below iterates over the returned list of all operations.


for op in circuit.circuit:
    print(op)

###############################################################################
# Note that the CNOT alias for the controlled NOT gate is labelled CX in the
# circuit. You can find all operation aliases in the source code and documentation
# for the operations module.

###############################################################################
# ## Simulating a circuit
