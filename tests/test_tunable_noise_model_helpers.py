# Copyright 2026 D-Wave
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

import pytest

from dwave.gate.utils.tunable_noise_model_helpers import (
    create_measurement_misassignment_noise,
    create_oneq_gate_noise,
    create_twoq_gate_noise,
)

X_PAULI_PROB = 0.11
Y_PAULI_PROB = 0.12
Z_PAULI_PROB = 0.13
LEAKAGE_PROB = 0.14
SEEPAGE_PROB = 0.15

MEAS_BITFLIP_PROB = 0.15
MEAS_LEAK_MISASSIGN = 0.2
MEAS_SEEP_MISASSIGN = 0.25


def test_tunable_model_helpers():
    noise_params = create_oneq_gate_noise(
        oneq_leakage=LEAKAGE_PROB,
        oneq_seepage=SEEPAGE_PROB,
        oneq_pauli_x=X_PAULI_PROB,
        oneq_pauli_y=Y_PAULI_PROB,
        oneq_pauli_z=Z_PAULI_PROB,
    )

    # Raise a value error if both depolarizing and individual pauli probabilities
    # are specified
    with pytest.raises(ValueError):
        create_oneq_gate_noise(
            oneq_leakage=LEAKAGE_PROB,
            oneq_seepage=SEEPAGE_PROB,
            oneq_depol=X_PAULI_PROB,
            oneq_pauli_x=X_PAULI_PROB,
            oneq_pauli_y=Y_PAULI_PROB,
            oneq_pauli_z=Z_PAULI_PROB,
        )

    depol = create_oneq_gate_noise(
        oneq_leakage=LEAKAGE_PROB,
        oneq_seepage=SEEPAGE_PROB,
        oneq_depol=X_PAULI_PROB,
    )

    assert depol["oneq_pauli_x"] == pytest.approx(X_PAULI_PROB / 3)
    assert depol["oneq_pauli_y"] == pytest.approx(X_PAULI_PROB / 3)
    assert depol["oneq_pauli_z"] == pytest.approx(X_PAULI_PROB / 3)

    # Raise a value error if both depolarizing and individual pauli probabilities
    # are specified
    with pytest.raises(ValueError):
        create_twoq_gate_noise(
            twoq_leakage=LEAKAGE_PROB,
            twoq_seepage=SEEPAGE_PROB,
            twoq_dephase=Z_PAULI_PROB,
            twoq_pauli_x=X_PAULI_PROB,
            twoq_pauli_y=Y_PAULI_PROB,
            twoq_pauli_z=Z_PAULI_PROB,
        )

    dephase = create_twoq_gate_noise(
        twoq_leakage=LEAKAGE_PROB,
        twoq_seepage=SEEPAGE_PROB,
        twoq_dephase=Z_PAULI_PROB,
    )

    assert dephase["twoq_pauli_x"] == pytest.approx(0.0)
    assert dephase["twoq_pauli_y"] == pytest.approx(0.0)
    assert dephase["twoq_pauli_z"] == pytest.approx(Z_PAULI_PROB)

    tmp_noise_params = create_twoq_gate_noise(
        twoq_leakage=LEAKAGE_PROB,
        twoq_seepage=SEEPAGE_PROB,
        twoq_pauli_x=X_PAULI_PROB,
        twoq_pauli_y=Y_PAULI_PROB,
        twoq_pauli_z=Z_PAULI_PROB,
    )

    noise_params = {**noise_params, **tmp_noise_params}

    tmp_noise_params = create_measurement_misassignment_noise(
        prob_bit_flip=MEAS_BITFLIP_PROB,
        prob_leak_misassign=MEAS_LEAK_MISASSIGN,
        prob_seep_misassign=MEAS_SEEP_MISASSIGN,
    )

    noise_params = {**noise_params, **tmp_noise_params}

    noise_params["prep_leakage"] = None
    noise_params["prep_bit_flip"] = X_PAULI_PROB

    assert noise_params["prep_leakage"] is None
    assert noise_params["twoq_leakage"] == LEAKAGE_PROB
    assert noise_params["twoq_seepage"] == SEEPAGE_PROB

    assert noise_params["oneq_leakage"] == pytest.approx(LEAKAGE_PROB)
    assert noise_params["oneq_seepage"] == pytest.approx(SEEPAGE_PROB)
    assert noise_params["oneq_pauli_x"] == pytest.approx(X_PAULI_PROB)
    assert noise_params["oneq_pauli_y"] == pytest.approx(Y_PAULI_PROB)
    assert noise_params["oneq_pauli_z"] == pytest.approx(Z_PAULI_PROB)

    assert noise_params["twoq_leakage"] == pytest.approx(LEAKAGE_PROB)
    assert noise_params["twoq_seepage"] == pytest.approx(SEEPAGE_PROB)
    assert noise_params["twoq_pauli_x"] == pytest.approx(X_PAULI_PROB)
    assert noise_params["twoq_pauli_y"] == pytest.approx(Y_PAULI_PROB)
    assert noise_params["twoq_pauli_z"] == pytest.approx(Z_PAULI_PROB)

    assert noise_params["meas_bit_flip"] == pytest.approx(MEAS_BITFLIP_PROB)
    assert noise_params["meas_leakage"] == pytest.approx(MEAS_LEAK_MISASSIGN)
    assert noise_params["meas_seepage"] == pytest.approx(MEAS_SEEP_MISASSIGN)
