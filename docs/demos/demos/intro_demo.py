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
construct, and simulate quantum circuits using a performant state-vector
simulator.

## Circuits and Operations
Begin with two necessary imports: The `Circuit` class, which will contain the
full quantum circuit with all the operations, measurements, qubits and bits, and
the `operations` module.
"""

from dwave.gate.circuit import Circuit
import dwave.gate.operations as ops

###############################################################################
# The operations, or gates, contain information related to a particular operation,
# including its matrix representation, potential decompositions, and how to apply
# it to a circuit.
#
# The `Circuit` keeps track of the operations and the logical
# flow of their executions. It also stores the qubits, bits and measurements.
#
# When initializing a circuit, you need to declare the number of qubits and,
# optionally, the number of bits (i.e., classical measurement results containers).
# You can add more qubits and bits using the `Circuit.add_qubit` and `Circuit.add_bit`
# methods.

circuit = Circuit(num_qubits=2, num_bits=2)

###############################################################################
# You can use the `operations` module (shortened above to `ops`) to access a
# variety of quantum gates; for example, the Pauli X operator. Note that you can
# instantiate an operation without declaring which qubits it should be applied
# to.

ops.X()

###############################################################################
# You can access the matrix property via either the operation class itself or an
# instance of the operation, which additionally can contain parameters and qubit
# information.

ops.X.matrix

###############################################################################
# If the matrix representation of an operator is dependent on parameters (e.g.,
# the rotation operation `ops.RX`) it can only be retrieved from an instance.

ops.RZ(4.2).matrix

###############################################################################
# ## The Circuit Context
#
# Operations are applied by calling either an operation class, or an instance of
# an operation class, within the context of the circuit, passing along the
# qubits on which the operation is applied.
#
# ```python
# with circuit.context:
#     # apply operations here
# ```
#

###############################################################################
# When you activate a context, a named tuple containing reference registers to
# the circuit's qubits and classical bits is returned. You can also access the
# qubit registers directly, via the `Circuit.qregisters` property, or the
# reference registers containing all the qubits, via `Circuit.qubits`.
#
# In the example below, the circuit contains a single qubit register with two
# qubits; it could contain any number of qubit registers. You can
#
# * add another register with the `Circuit.add_qregister` method, where an argument `n`
#   is the number of qubits in the new register.
# * add a qubit with `Circuit.add_qubit`, optionally passing a qubit object
#   and/or a register to which to add it.

circuit = Circuit(2)

with circuit.context as reg:
    ops.X(reg.q[0])

###############################################################################
# This has created a circuit object with two qubits in its register, applying
# a single X gate to the first qubit. Print the circuit to see general information
# about it: type of circuit, number of qubits/bits, and number of operations.

print(circuit)

###############################################################################
# ## Applying Gates to Circuits
#
# You can apply operations to a circuit in several different ways, as demonstrated
# in the example below. You can pass both qubits and parameters as either single
# values (when supported by the gate) or as sequences. Note that different types of
# gates accept slightly different arguments, although you can _always_ pass the
# qubits as sequences via the keyword argument `qubits`.
#
# :::note
# Always apply any operations you instantiate within a circuit context to
# specific qubits in the circuit's qubit register. You can access the qubit
# register via the named tuple, returned by the context manager as `q`, indexing
# into it to retrieve the corresponding qubit.
# :::

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
# You can access all the operations in a circuit using the `Circuit.circuit`
# property. The code below iterates over a list of all operations that
# have been applied to the circuit and prints each one separately.

for op in circuit.circuit:
    print(op)

###############################################################################
# :::note
# The CNOT alias for the controlled NOT gate is labelled CX in the circuit. You
# can find all operation aliases in the source code and documentation for the
# operations module.
# :::

###############################################################################
# ## Simulating a Circuit
#
# `dwave-gate` comes with a performant state-vector simulator. You can call it by
# passing a circuit to the `simulate` method, which will update the quantum state
# stored in the circuit, accessible via `Circuit.state`.

from dwave.gate.simulator import simulate

###############################################################################
# The following example creates a circuit object with 2 qubits and 1 bit in a
# quantum and a classical register respectively---the bit is required to store a
# single qubit measurement---and then applies a Hadamard gate and a CNOT gate to
# the circuit.

circuit = Circuit(2, 1)

with circuit.context as (q, c):
    ops.Hadamard(q[0])
    ops.CNOT(q[0], q[1])

###############################################################################
# You can now simulate the circuit, updating its stored quantum state.

simulate(circuit)

###############################################################################
# Printing the state reveals the expected state-vector $\frac{1}{\sqrt{2}}\left[1,
# 0, 0, 1\right]$ corresponding to the state:
#
# $$\vert\psi\rangle = \frac{1}{\sqrt{2}}\left(\vert00\rangle + \vert11\rangle\right)$$

circuit.state

###############################################################################
# ## Measurements
#
# Measurements work like any other operation in `dwave-gate`. The main difference
# is that the operation generates a measurement value when simulated, which can be
# stored in the classical register by piping it into a classical bit.
#
# The circuit above can be reused by simply unlocking it and appending a `Measurement` to it.

circuit.unlock()
with circuit.context as (q, c):
    m = ops.Measurement(q[1]) | c[0]

###############################################################################
# :::note
# This example stored the measurement instance as `m`, which you can use for
# post-processing. It's also possible to do this with all other operations in
# the same way, allowing for multiple identical operation applications.
#
# ```python
# with circuit.context as q, _:
#     single_x_op = ops.X(q[0])
#     # apply the X-gate again to the second qubit
#     # using the previously stored operation
#     single_x_op(q[1])
# ```
#
# This procedure can also be shortened into a single line for further convenience.
#
# ```python
# ops.CNOT(q[0], q[1])(q[1], q[2])(q[2], q[3])
# ```
# :::

###############################################################################
# The circuit should now contain 3 operations: a Hadamard, a CNOT and a measurement.

print(circuit)

###############################################################################
# When simulating this circuit, the measurement is applied and the measured
# value is stored in the classical register. Since a measurement affects the
# quantum state, the resulting state has collapsed into the expected result
# dependent on the value which has been measured.

simulate(circuit)

###############################################################################
# If the measurement result is 0 the state should have collapsed into $\vert00\rangle$,
# and if the measurement result is 1 the state should have collapsed into $\vert11\rangle$.
# Outputting the measurement value and the state reveals that this is indeed the case.

print(circuit.bits[0].value)
print(circuit.state)

###############################################################################
# ## Measurement Post-Access
# Since the measurement operation has been stored in `m`, you can use it to access the
# state as it was before the measurement.
#
# :::note
# Accessing the state of the circuit, along with any measurement post-sampling and
# state-access, is only available for simulators.
# :::

m.state

###############################################################################
# You can also sample that same state again using the `Measurement.sample` method,
# which by default only samples the state once. Here, request 10 samples.

m.sample(num_samples=10)

###############################################################################
# Finally, you can calculate the expected value of the measurement based on a
# specific number of samples.

m.expval(num_samples=10000)

""

