# Copyright 2023 D-Wave Systems Inc.
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

"""QIR circuit loader.

Contains the QIR-to-dwave-gate circuit object loaders.
"""

from __future__ import annotations

__all__ = [
    "load_qir_bitcode",
    "load_qir_string",
]

import re
from typing import Optional, Sequence, Tuple

import pyqir
from pyqir import Context, Module, Opcode

import dwave.gate.operations as ops
from dwave.gate import Circuit
from dwave.gate.qir.instructions import InstrType, Operations


def load_qir_bitcode(
    qir_bitcode: bytes, circuit: Optional[Circuit] = None, context: Context = None
) -> Circuit:
    """Loads QIR bitcode into a ``Circuit`` object.

    Args:
        qir_bitcode: QIR bitcode representation of the circuit.
        circuit: Optional circuit into which to load QIR bitcode. If ``None``
            a new circuit is created (default).
        context: Context to use to construct the module.

    Returns:
        Circuit: Circuit representation of the QIR.
    """
    module = Module.from_bitcode(context or Context(), qir_bitcode)
    return _module_to_circuit(module, circuit)


def load_qir_string(
    qir_str: str, circuit: Optional[Circuit] = None, context: Context = None
) -> Circuit:
    """Loads a QIR string into a ``Circuit`` object.

    Args:
        qir_str: QIR string representation of the circuit.
        circuit: Optional circuit into which to load QIR bitcode. If ``None``
            a new circuit is created (default).
        context: Context to use to construct the module.

    Returns:
        Circuit: Circuit representation of the QIR.
    """
    module = Module.from_ir(context or Context(), qir_str)
    return _module_to_circuit(module, circuit)


def _module_to_circuit(module: Module, circuit: Optional[Circuit] = None) -> Circuit:
    """Parses a PyQIR module into a ``Circuit`` object.

    Args:
        module: PyQIR module containing the QIR representation of the circuit.
        circuit: Optional circuit into which to load QIR bitcode. If ``None``
            a new circuit is created (default).

    Returns:
        Circuit: Circuit representation of the QIR.
    """
    if circuit is None:
        circuit = Circuit()

    qubits = []

    meas_qubits = []
    meas_bits = []

    for func in module.functions:
        ret_set = False
        for block in func.basic_blocks:
            # if return has been set, then ignore rest of blocks
            if ret_set:
                break

            for instr in block.instructions:
                if instr.opcode == Opcode.RET:
                    # break block on return code
                    ret_set = True
                    break

                if instr.opcode == Opcode.CALL:
                    # construct and add operations to circuit
                    op_name = Operations.from_qis_operation.get(instr.callee.name, None)
                    if "__qis__" not in instr.callee.name:
                        # only add non-ignored QIS operations
                        continue
                    elif op_name is None:
                        raise TypeError(f"'{instr.callee.name}' not found in valid QIS operations.")

                    params, qubits, bits = _deconstruct_call_instruction(instr)

                    if op_name == "Measurement":
                        meas_qubits.extend(qubits)
                        meas_bits.extend(bits)
                        # delay adding until all measurements are collected
                        continue

                    for _ in range(circuit.num_qubits, max(qubits or [0]) + 1):
                        circuit.add_qubit()

                    circuit.unlock()
                    with circuit.context as reg:
                        getattr(ops, op_name)(*params, qubits=[reg.q[qubit] for qubit in qubits])

    # finally, add measurements (if any) to circuit
    if meas_qubits:
        _add_measurements(circuit, meas_qubits, meas_bits)

    return circuit


def _add_measurements(circuit: Circuit, qubits: Sequence[int], bits: Sequence[int]) -> None:
    """Add measurements to circuit.

    Args:
        circuit: The circuit to add the measurments to.
        qubits: The qubits that are being measured.
        bits: The bits in which the measurement values are stored.
    """
    for _ in range(circuit.num_bits, max(bits or [0]) + 1):
        circuit.add_bit()

    circuit.unlock()
    with circuit.context as reg:
        qubits = [reg.q[qubit] for qubit in qubits]
        bits = [reg.c[bit] for bit in bits]

        ops.Measurement(qubits=qubits) | bits


def _deconstruct_call_instruction(
    instr: pyqir.Instruction,
) -> Tuple[Sequence[float], Sequence[int]]:
    """Extracts parameters and qubits from a call instruction.

    Args:
        instr: PyQIR instruction to deconstruct.

    Returns:
        tuple: Containing the parameters and qubits of the instruction/operation.
    """
    params = []
    qubits = []
    bits = []
    for arg in instr.args:
        # only supports doubles
        if arg.type.is_double:
            params.append(arg.value)
        elif arg.type.pointee.name == "Qubit":
            if arg.is_null:
                qubits.append(0)
            else:
                pattern = "\%Qubit\* inttoptr \(i64 (\d+) to \%Qubit\*\)"
                qubit = re.search(pattern, str(arg)).groups()[0]
                qubits.append(int(qubit) if qubit.isdigit() else None)
        else:
            assert arg.type.pointee.name == "Result"
            if arg.is_null:
                bits.append(0)
            else:
                pattern = "\%Result\* inttoptr \(i64 (\d+) to \%Result\*\)"
                bit = re.search(pattern, arg.__str__()).groups()[0]
                bits.append(int(bit) if bit.isdigit() else None)

    return params, qubits, bits
