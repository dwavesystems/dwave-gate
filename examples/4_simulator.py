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

"""Running circuits on the simulator."""

import numpy as np
np.set_printoptions(suppress=True)

import dwave.gate.operations as ops

from dwave.gate.circuit import Circuit
from dwave.gate.simulator.simulator import simulate

# Let's build a simple circuit that applies an X gate to the first qubit, and then a CNOT gate on
# both qubits, which will deterministically also apply an X gate to the second qubit since it's
# controlled by the first.
circuit = Circuit(2)

with circuit.context as (q, c):
    ops.X(q[0])
    ops.CNOT(q[0], q[1])

print("Circuit", circuit.circuit)

# We can now simulate this circuit and print the circuits state-vector.
simulate(circuit)

# Note that it returns the vector representing the state |11>, which is expected.
print(circuit.state)

# Using the tools and operations explained in the previous examples, arbitrary circuits can be
# constructed and simulated, returning the final state. Note also that the state is always
# initialized to all 0 (e.g., in the case above, |00>). This can be altered by applying the gates
# necessary to create the wanted state.
