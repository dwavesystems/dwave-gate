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

import numpy as np
import pytest

import dwave.gate.operations as ops
from dwave.gate.circuit import Circuit, CircuitError
from dwave.gate.simulator.simulator import simulate


class TestSimulateMeasurements:
    """Unit tests for simulating (mid-circuit) measurements."""

    def test_measurements(self):
        """Test simulating a circuit with a measurement."""
        circuit = Circuit(1, 1)

        with circuit.context as (q, c):
            ops.X(q[0])
            m = ops.Measurement(q[0]) | c[0]

        _ = simulate(circuit)

        assert m.bits
        assert m.bits[0] == c[0]
        assert c[0] == 1

    def test_measurements_little_endian(self):
        """Test that the correct error is raised when measuring using little-endian notation."""
        circuit = Circuit(1, 1)

        with circuit.context as (q, c):
            ops.X(q[0])
            ops.Measurement(q[0]) | c[0]
        with pytest.raises(CircuitError, match="only supports big-endian"):
            _ = simulate(circuit, little_endian=True)

    def test_measurement_state(self):
        """Test accessing a state from a measurement."""
        circuit = Circuit(1, 1)

        with circuit.context as (q, c):
            ops.X(q[0])
            m = ops.Measurement(q[0]) | c[0]

        _ = simulate(circuit)

        assert np.all(m.state == np.array([0, 1]))

    @pytest.mark.parametrize("n", [0, 1, 100])
    def test_measurement_sample(self, n):
        """Test sampling from a measurement."""
        circuit = Circuit(1, 1)

        with circuit.context as (q, c):
            ops.Hadamard(q[0])
            m = ops.Measurement(q[0]) | c[0]

        _ = simulate(circuit)

        samples = m.sample(num_samples=n) or []
        assert len(samples) == n
        assert all([i in (0, 1) for i in samples])

    def test_measurement_sample_multiple_qubits(self):
        """Test sampling from a measurement on multiple qubits."""
        circuit = Circuit(2, 2)

        with circuit.context as (q, c):
            ops.X(q[0])
            m = ops.Measurement(q) | c

        _ = simulate(circuit)

        with pytest.raises(ValueError, match="Must specify which to sample"):
            _ = m.sample()

        assert m.sample(0) == [1]
        assert m.sample(1) == [0]

    def test_measurement_sample_nonexistent_qubit(self):
        """Test sampling from a measurement on a non-existent qubit."""
        circuit = Circuit(1, 1)

        with circuit.context as (q, c):
            ops.X(q[0])
            m = ops.Measurement(q) | c

        _ = simulate(circuit)

        assert m.sample(0) == [1]
        with pytest.raises(ValueError, match="Cannot sample qubit"):
            assert m.sample(1) == [0]

    def test_measurement_expval(self):
        """Test measuring an expectation value from a measurement."""
        circuit = Circuit(1, 1)

        with circuit.context as (q, c):
            ops.Hadamard(q[0])
            m = ops.Measurement(q[0]) | c[0]

        _ = simulate(circuit)
        # expectation values are random; assert that it's between 0.4 and 0.6
        assert 0.5 == pytest.approx(m.expval(), 0.2)

    def test_measurement_no_simulation(self):
        """Test a circuit with a measurement without simulating it."""
        circuit = Circuit(1, 1)

        with circuit.context as (q, c):
            ops.Hadamard(q[0])
            m = ops.Measurement(q[0])

        assert m.expval() is None
        assert m.sample() is None
        assert m.state is None


class TestConditionalOps:
    """Unit tests for running conditional operations on the simulator."""

    def test_conditional_op_true(self):
        """Test simulating a circuit with a conditional op (true)."""
        circuit = Circuit(2, 1)

        with circuit.context as (q, c):
            ops.X(q[0])
            ops.Measurement(q[0]) | c[0]  # state is |10>
            ops.X(q[1]).conditional(c[0])

        res = simulate(circuit)
        # should apply X on qubit 1 changing state to |11>
        expected = np.array([0, 0, 0, 1])

        assert np.allclose(res, expected)

    def test_conditional_op_false(self):
        """Test simulating a circuit with a conditional op (false)."""
        circuit = Circuit(2, 1)

        with circuit.context as (q, c):
            ops.Measurement(q[0]) | c[0]  # state is |00>
            ops.X(q[1]).conditional(c[0])

        res = simulate(circuit)
        # should NOT apply X on qubit 1 leaving state in |00>
        expected = np.array([1, 0, 0, 0])

        assert np.allclose(res, expected)

    def test_conditional_op_multiple_qubits_false(self):
        """Test simulating a circuit with a multiple conditional ops (false)."""
        circuit = Circuit(2, 2)

        with circuit.context as (q, c):
            ops.X(q[0])
            ops.Measurement(q) | c  # state is |10>
            x = ops.X(q[1]).conditional(c)

        res = simulate(circuit)
        # should NOT apply X on qubit 1 leaving state in |10>
        expected = np.array([0, 0, 1, 0])

        assert np.allclose(res, expected)
        assert x.is_blocked

    def test_conditional_op_multiple_qubits_true(self):
        """Test simulating a circuit with a multiple conditional ops (true)."""
        circuit = Circuit(2, 2)

        with circuit.context as (q, c):
            ops.X(q[0])
            ops.X(q[1])
            ops.Measurement(q) | c  # state is |11>
            x = ops.X(q[0]).conditional(c)

        res = simulate(circuit)
        # should apply X on qubit 0 changing state to |01>
        expected = np.array([0, 1, 0, 0])

        assert np.allclose(res, expected)
        assert not x.is_blocked

    def test_bell_state_measurement(self):
        """Test that measurement gives one of two correct states and sets the resulting
        state vector correctly for the Bell state."""

        circuit = Circuit(2, 2)

        with circuit.context as (q, c):
            ops.Hadamard(q[0])
            ops.CNOT(q[0], q[1])
            ops.Measurement(q) | c

        res = simulate(circuit)

        measurement = tuple(b.value for b in circuit.bits)

        if measurement == (0, 0):
            assert np.allclose(res, [1, 0, 0, 0])
        elif measurement == (1, 1):
            assert np.allclose(res, [0, 0, 0, 1])
        else:
            assert False

    def test_non_entangled_measurement(self):
        """Test single qubit measurement is correct on after Hadamards."""
        circuit = Circuit(2, 1)

        with circuit.context as (q, c):
            ops.Hadamard(q[0])
            ops.Hadamard(q[1])
            ops.Measurement(q[0]) | c[0]

        res = simulate(circuit)

        if circuit.bits[0].value == 0:
            assert np.allclose(res, [1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        else:
            assert np.allclose(res, [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])

    def test_measurement_rng_seed(self):
        """Test measurement is reproducible after setting RNG seed."""
        num_qubits = 6
        circuit = Circuit(num_qubits, num_qubits)

        with circuit.context as (q, c):
            for i in range(num_qubits):
                ops.Hadamard(q[i])
            ops.Measurement(q) | c

        simulate(circuit, rng_seed=666)
        expected = tuple(b.value for b in circuit.bits)

        for _ in range(100):
            for bit in circuit.bits:
                bit.reset()
            simulate(circuit, rng_seed=666)
            assert expected == tuple(b.value for b in circuit.bits)
