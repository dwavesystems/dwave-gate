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


def create_oneq_gate_noise(
    oneq_leakage: float | None = None,
    oneq_seepage: float | None = None,
    oneq_depol: float | None = None,
    oneq_pauli_x: float | None = None,
    oneq_pauli_y: float | None = None,
    oneq_pauli_z: float | None = None,
) -> dict[str, float | None]:
    """Create noise parameters for single-qubit gates for a tunable noise model.

    Args:
        oneq_leakage: Probability of a leakage occurring during a one qubit gate. 
            Defaults to None.
        oneq_seepage: Probability of a seepage occurring during a one qubit gate. 
            Defaults to None.
        oneq_depol: Overall one qubit gate depolarizing error probability.
            Defaults to None.
        oneq_pauli_x: Probability of an X Pauli error occurring during an one qubit gate. 
            Defaults to None. If oneq_depol is given, will be oneq_depol/3.
        oneq_pauli_y: Probability of an Y Pauli error occurring during an one qubit gate. 
            Defaults to None. If oneq_depol is given, will be oneq_depol/3.
        oneq_pauli_z: Probability of an Z Pauli error occurring during an one qubit gate.
            Defaults to None. If oneq_depol is given, will be oneq_depol/3.

    Raises:
        ValueError: If user tries to specify both an overall depolarizing error
            and individual pauli errors.

    Returns:
        The noise param dict.
    """

    if oneq_depol is not None and (
        oneq_pauli_x is not None or oneq_pauli_y is not None or oneq_pauli_z is not None
    ):
        raise ValueError(
            "Cannot specify an overall one qubit gate depolarizing error"
            " and individual Pauli errors."
        )

    if oneq_depol is not None:
        oneq_pauli_x = oneq_depol / 3
        oneq_pauli_y = oneq_depol / 3
        oneq_pauli_z = oneq_depol / 3

    noise_param_dict = dict()

    noise_param_dict["oneq_leakage"] = oneq_leakage
    noise_param_dict["oneq_seepage"] = oneq_seepage
    noise_param_dict["oneq_pauli_x"] = oneq_pauli_x
    noise_param_dict["oneq_pauli_y"] = oneq_pauli_y
    noise_param_dict["oneq_pauli_z"] = oneq_pauli_z

    return noise_param_dict


def create_twoq_gate_noise(
    twoq_leakage: float | None = None,
    twoq_seepage: float | None = None,
    twoq_dephase: float | None = None,
    twoq_pauli_x: float | None = None,
    twoq_pauli_y: float | None = None,
    twoq_pauli_z: float | None = None,
) -> dict[str, float | None]:
    """Create noise parameters for two-qubit entangling gates for a tunable noise model.

    Args:
        twoq_leakage: Probability of a leakage occurring during a two qubit gate.
            Defaults to None.
        twoq_seepage: Probability of a seepage occurring during a two qubit gate. 
            Defaults to None.
        twoq_dephase: Overall dephasing probability of a two qubit gate.
            Defaults to None.
        twoq_pauli_x: Probability of an X Pauli error occurring during a two
            qubit gate. Defaults to None. If twoq_dephase is given, this is 0.0.
        twoq_pauli_y: Probability of an Y Pauli error occurring during a two
            qubit gate. Defaults to None. If twoq_dephase is given, this is 0.0.
        twoq_pauli_z: Probability of an Z Pauli error occurring during a two
            qubit gate. Defaults to None. If twoq_dephase is given, this is 0.0.

    Returns:
        The updated (or newly created) noise param dict.
    """

    if twoq_dephase is not None and (
        twoq_pauli_x is not None or twoq_pauli_y is not None or twoq_pauli_z is not None
    ):
        raise ValueError(
            "Cannot specify an overall two qubit gate dephasing error"
            " and individual Pauli errors."
        )

    if twoq_dephase is not None:
        twoq_pauli_x = 0.0
        twoq_pauli_y = 0.0
        twoq_pauli_z = twoq_dephase

    noise_param_dict = dict()

    noise_param_dict["twoq_leakage"] = twoq_leakage
    noise_param_dict["twoq_seepage"] = twoq_seepage
    noise_param_dict["twoq_pauli_x"] = twoq_pauli_x
    noise_param_dict["twoq_pauli_y"] = twoq_pauli_y
    noise_param_dict["twoq_pauli_z"] = twoq_pauli_z

    return noise_param_dict


def create_measurement_misassignment_noise(
    prob_bit_flip: float | None = None,
    prob_leak_misassign: float | None = None,
    prob_seep_misassign: float | None = None,
) -> dict[str, float | None]:
    """Create noise parameters for measurements for a tunable noise model.

    Args:
        prob_bit_flip: Probability of measuring 1 when qubit is in 0 or vice versa.
        prob_leak_misassign: Probability of measuring a "*" when qubit is
            actually in a logical state.
        prob_seep_misassign: Probability of measuring 0 or 1 when qubit is
            actually leaked.

    Returns:
        The updated (or newly created) noise param dict.
    """
    noise_param_dict = dict()

    noise_param_dict["meas_bit_flip"] = prob_bit_flip
    noise_param_dict["meas_leakage"] = prob_leak_misassign
    noise_param_dict["meas_seepage"] = prob_seep_misassign
    return noise_param_dict
