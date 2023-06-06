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
# Beginner's guide to `dwave-gate`

With `dwave-gate` you can easily contruct and simulate quantum circuits. This tutorial will guide
you through how to use the `dwave-gate` library to inspect different quantum gates, construct
circuits out of them, and then simulate them using our performant state-vector simulator.

We begin by importing the necessary modules.
"""

from dwave.gate.circuit import Circuit
import dwave.gate.operations as ops

###############################################################################
# The `Circuit` object contains the full quantum circuit, with all the operations, measurements,
# qubits and bits. Operations, or gates, are objects that contain information related to that
# specific operation, including its matrix representation, potential decompositions, and how to
# apply it to a circuit. Via the `operations` module, here shortened to `ops`, a variety of quantum
# gates can be accessed.

ops.X()

###############################################################################
# As we can see, any operation can be instantiated without declaring which qubits it should be
# applied to. We can also call the matrix property on either the gate class itself or on an instance
# (i.e., an instantiated operation, which can contain additional parameters and/or qubit
# information).

ops.X.matrix

###############################################################################
# If the matrix representation is dependent on parameters (e.g., the X-rotation operation) it can
# only be retrieved from an instance.


ops.RZ(4.2).matrix

###############################################################################
# Operations are applied when either the class, or an instance of the class, is called within a
# circuit's context. They can be applied to the circuit in several different ways, as detailed
# below. Both qubits and parameters can be passed either as single values (if supported by the gate)
# or as sequences. Note that different types of gates accept slightly different argument, although
# the qubits can _always_ be passed as sequences via the keyword argument `qubits`.
#
# When instantiated within a circuit context, operations must always be applied to specific qubits
# which in turn must be part of the circuits qubit register. The qubit register can be accessed via
# the named tuple returned by the context manager as `q`, indexing into it to retrieve the
# corresponding qubit.
#
# Let's start by creating a circuit object with 2 qubits in its register, and apply a single X-gate
# to the first qubit.

circuit = Circuit(2)

with circuit.context as reg:
    ops.X(reg.q[0])

###############################################################################
# We can access the qubit register via the `Circuit.qregisters` attribute, which currently should
# contain a single qubit register with 2 qubits in it, but could contain any number of qubit
# registers. If we'd want to, we could add another register with the `Circuit.add_qregister(n)`
# method, where `n` would be the number of qubits in the new register, or add a qubit with
# `Circuit.add_qubit()`, optinally passing a qubit object and/or a register to which to add it.
#
# The registers tuple can also be unwrapped directly into a qubit register `q` (and a classical
# register `c`, but we'll get into that later).
#
# Below follows a few examples for how to apply different gates to the circuit.


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
# Printing the above circuit gives us some general information about the circuit: the type of
# circuit, the number of qubits/bits and the number of operations.

print(circuit)

###############################################################################
# We can also access the operations in the circuit via the `Circuit.circuit` attribute. This will
# return a list of all operations which we can iterate over and print in the console.


for op in circuit.circuit:
    print(op)

###############################################################################
# Note that e.g., CNOT and CX apply the exact same gate. CNOT is only an alias for CX (so they both
# are labelled as CX in the circuit). There are also other aliases which you can spot either in the
# source code for the operations module or read more about in the documentation.
#

###############################################################################
# ## Simulating a circuit

""
