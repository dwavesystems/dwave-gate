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

"""Measuring circuits on the simulator."""

import numpy as np
np.set_printoptions(suppress=True)

import dwave.gate.operations as ops

from dwave.gate.circuit import Circuit
from dwave.gate.simulator.simulator import simulate

# Let's again build a simple circuit that applies a Hadamard gate to the first qubit and then a CNOT gate
# on both qubits, but this time we measure the second qubit instead of simply returning the state.

# To store the measured value, a classical register is required.
circuit = Circuit(2, 1)

# The classical register can be accessed via the context manager, commonly denoted 'c'. The
# measurement can then be piped over to the classical register and stored in a bit object.
with circuit.context as (q, c):
    ops.Hadamard(q[0])
    ops.CNOT(q[0], q[1])
    m = ops.Measurement(q[1]) | c[0]

print("Circuit", circuit.circuit)

# We can simulate this circuit as usual which will return it's state-vector, but due to the
# measurement the state will collapse into a substate. The resulting state after running the above
# circuit depends on the measurement outcome since the state before the measurment is entangled.
simulate(circuit)
print(circuit.state)

# We can easily check the classical register, and the containing bits, to see which value was
# measured. Since we measured the first bit in the register, we choose it in the register.
print(circuit.bits[0].value)

# Since we've stored the measurement operation in 'm' we can use it to check the measured
# qubits, bit values and the state at measurement.
print(f"Bits: {m.bits}")
print(f"State: {m.state}")

# The measurement operation also allows for further sampling...
print(m.sample())

# ...and expectation value calculations.
print(m.expval())

# Note that the state, sampling and expectation values are only accessible on simulators and cannot
# be retrieved if the circuit would be run on quantum hardware.