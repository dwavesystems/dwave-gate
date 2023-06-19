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

"""Core QIR compiler.

Contains the core QIR compiler which uses the PyQIR API.
"""

from __future__ import annotations

__all__ = [
    "BaseModule",
    "qir_module",
]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import pyqir
from pyqir import BasicBlock, Context, Linkage
from pyqir.rt import initialize

if TYPE_CHECKING:
    from dwave.gate import Circuit

# from dwave.gate.operations.base import ParametricOperation
from dwave.gate.qir.instructions import InstrType, Instruction, Operations


class CompileError(Exception):
    """Exception to be raised when there is an error with QIR compilation."""


class BaseModule:
    """Module representing a QIR program conforming to the QIR basic profile.

    Any keyword arguments are directly passed to the internal PyQIR Module.
    Below are listed the required keyword arguments with their defualt values.

    Args:
        name: Name of the module.
        num_qubit: Number of qubits that the module contains.
        num_bits: Number of bits/results that the module contains.
        context: The context which the module is tied to. If ``None`` a new context is created.

    Keyword args:
        qir_major_version: The QIR major version. Defaults to 1.
        qir_minor_version: The QIR minor version. Defaults to 0.
        dynamic_qubit_management: Whether to use dynamic qubit management. Defaults to ``False``.
        dynamic_result_management: Whether to use dynamic result management. Defaults to ``False``.
    """

    def __init__(
        self,
        name: str,
        num_qubits: int,
        num_bits: int,
        context: Optional[pyqir.Context] = None,
        **kwargs,
    ) -> None:
        self._is_compiled = False

        self._module_kwargs = kwargs
        self._module_kwargs.setdefault("qir_major_version", 1)
        self._module_kwargs.setdefault("qir_minor_version", 0)
        self._module_kwargs.setdefault("dynamic_qubit_management", False)
        self._module_kwargs.setdefault("dynamic_result_management", False)

        self._module = pyqir.qir_module(context or Context(), name, **self._module_kwargs)

        self._builder = pyqir.Builder(self.context)
        self._num_qubits = num_qubits
        self._num_bits = num_bits

        self._qubits = [pyqir.qubit(self.context, idx) for idx in range(num_qubits)]
        self._results = [pyqir.result(self.context, idx) for idx in range(num_bits)]

        self._return = None
        self._return_cmd = None

        # the following keys are determined by the QIR Base Profile
        self._instructions = {
            "body": [],
            "measurements": [],
            "output": [],
        }

        self._external_functions: Dict[str, BaseModule.Types] = {}

        @dataclass
        class Types:
            """Dataclass for containing QIR types required for external functions.

            Consolidates the use of types instead of relying on pyqir structure and ties the
            relevant types to the current module, providing a simple type interface.
            """

            # types that only require the module context
            VOID = pyqir.Type.void(self.context)
            DOUBLE = pyqir.Type.double(self.context)
            QUBIT = pyqir.qubit_type(self.context)
            RESULT = pyqir.result_type(self.context)

            # integer and boolean types
            INT32 = pyqir.IntType(self.context, 32)
            INT64 = pyqir.IntType(self.context, 64)
            INT_ = INT64

            INT1 = pyqir.IntType(self.context, 1)
            BOOL_ = INT1

            # types that require arguments
            function = pyqir.FunctionType
            pointer = pyqir.PointerType

            # types without constructors
            ARRAY = pyqir.ArrayType
            STRUCT = pyqir.StructType

        self.Types = Types

    def compile(self, force: bool = False, strict: bool = True) -> None:
        """Compile the module and verify it.

        Args:
            force: Whether to force recompilation if already compiled.
            set_void_return: Whether to set a ``Void`` return in the final block.
            strict: Whether the compiler should raise an error if verification fails.
        """
        if self._is_compiled:
            if not force:
                raise CompileError(
                    "Module already compiled. Set 'force=True' to force recompilation."
                )
            self.reset(rm_instructions=False)

        self._compile()

        self._return = self.builder.ret(self._return_cmd)

        error_msg = self._module.verify()
        if error_msg is not None and strict:
            raise CompileError(error_msg)

        self._qir = str(self._module)
        self._bitcode = self._module.bitcode

        self._is_compiled = True

    def _compile(self) -> None:
        """Construct the module by adding instructions and blocks."""
        entry_point = pyqir.entry_point(self._module, "main", self._num_qubits, self._num_bits)

        block = BasicBlock(self.context, "entry", parent=entry_point)
        self.builder.insert_at_end(block)

        int_type_pointer = pyqir.PointerType(pyqir.IntType(self.context, 8))
        null_const = pyqir.Constant.null(int_type_pointer)
        initialize(self.builder, null_const)

        for block_name, instructions in self._instructions.items():
            # connect to next block
            block = BasicBlock(self.context, block_name, parent=entry_point)
            self.builder.br(block)
            self.builder.insert_at_end(block)

            # add operations
            for instr in instructions:
                if block_name == "output":
                    pyqir.rt.result_record_output(self.builder, instr.args[0], null_const)
                else:
                    instr.execute(self.builder)

    @property
    def name(self) -> str:
        """Module source filename."""
        return self._module.source_filename

    @property
    def qubits(self) -> Sequence[pyqir.Constant]:
        """QIR module qubit references."""
        return self._qubits

    @property
    def results(self) -> Sequence[pyqir.Constant]:
        """QIR module results/bit references."""
        return self._results

    @property
    def context(self) -> pyqir.Context:
        """Module context."""
        return self._module.context

    @property
    def builder(self) -> pyqir.Builder:
        """Module builder."""
        return self._builder

    def get_external_instruction(self, name: str, args: Sequence[pyqir.Value]) -> Instruction:
        """Gets an external function instruction.

        Args:
            name: The name of the function. Usually lowercase, QIR style,
                e.g., ``"x"`` or ``cnot``.
            args: The arguments for applying the function.

        Returns:
            Optional[pyqir.Function]: Returns the stored external function
            or ``None`` if not found.
        """
        external = self._external_functions.get(name, None)

        return Instruction(InstrType.EXTERNAL, name, args, external)

    @property
    def qir(self) -> str:
        """QIR string representation of the circuit.

        Will be compiled if not previously compiled manually.

        Returns:
            str: QIR string representation.
        """
        if not self._is_compiled:
            self.compile()

        return self._qir

    @property
    def bitcode(self) -> bytes:
        """Bitcode representation of the circuit.

        Will be compiled if not previously compiled manually.

        Returns:
            bytes: A bitcode representation of the QIR circuit.
        """
        if not self._is_compiled:
            self.compile()

        return self._module.bitcode

    def add_instruction(self, block: str, instruction: Instruction) -> None:
        """Add an instruction (operation) to the module.

        Args:
            block: The name of the block that the instruction should be added to.
            instruction: The instruction that should be added.
        """
        self._instructions[block].append(instruction)

    def add_external_function(
        self, name: str, parameter_types: Sequence[BaseModule.Types], return_type=None
    ) -> None:
        """Add an external function to the module.

        Args:
            name: The name of the external function. Preferably lowercase
                QIR style, e.g., ``"x"`` or ``"cnot"``.
            parameter_types: Sequence of parameter types to the QIR function.
            return_type: Return type of the QIR function. If ``None`` (default),
                return type is set to ``void``.
        """
        type_ = self.Types.function(return_type or self.Types.VOID, parameter_types)
        qis_name = f"__quantum__qis__{name}__body"
        self._external_functions[name] = pyqir.Function(
            type_, Linkage.EXTERNAL, qis_name, self._module
        )

    def set_return(self, ret_cmd: Optional[pyqir.Type.Value] = None, force: bool = False) -> None:
        """Set the return instruction.

        Will place the return instruction at the end of the final block.
        If forcing a re-set, the new return will again be placed at the
        end of the latest block, with the old one removed.

        Args:
            ret_cmd: The value to return. Returns void if ``None``.
            force: Whether to replace the former return command with a new one.
        """
        if self._return_cmd and not force:
            raise ValueError(
                f"Return {self._return_cmd} already set. Replace by passing 'force=True'"
            )
        self._return_cmd = ret_cmd

    def reset(self, rm_instructions: bool = True) -> None:
        """Reset the module for recompilation.

        Args:
            rm_instructions: Whether to also reset all instructions.
        """
        self._module = pyqir.qir_module(self.context, self.name, **self._module_kwargs)
        self._remove_external_instructions()
        self._return.erase()

        if rm_instructions:
            for key in self._instructions:
                self._instructions[key] = []

            self._return_cmd = None

            self._external_functions.clear()
        else:
            # replace all external function with new ones using the _new_ module
            # must be after the new 'self._module' declaration above
            for name, exfunc in self._external_functions.items():
                function_type = self.Types.function(
                    exfunc.type.ret, [p.type for p in exfunc.params]
                )
                self._external_functions[name] = pyqir.Function(
                    function_type, Linkage.EXTERNAL, exfunc.name, self._module
                )

        self._is_compiled = False
        self._qir = self._bitcode = None

    def _remove_external_instructions(self) -> None:
        """Removes all external functions from the instructions."""
        for block, instructions in self._instructions.items():
            instr_len = len(instructions)
            for i, instr in enumerate(reversed(instructions)):
                if instr.type is InstrType.EXTERNAL:
                    del self._instructions[block][instr_len - i - 1]


def qir_module(circuit: Circuit, add_external: bool = False) -> BaseModule:
    """Create a QIR module out of a circuit.

    Args:
        circuit: Circuit to convert to a QIR module.
        add_external: Whether to add external function that are not part of the QIR Base Profile.

    Returns:
        BaseModule: QIR module representing the circuit.
    """
    module = BaseModule("Circuit", num_qubits=circuit.num_qubits, num_bits=circuit.num_bits)

    block = "body"
    for op in circuit.circuit:
        type_, pyqir_op = Operations.to_qir.get(op.__class__.name, (None, None))
        if add_external and type_ is InstrType.EXTERNAL:
            qubit_type = module.Types.QUBIT
            param_type = module.Types.DOUBLE

            num_parameters = getattr(op, "num_parameters", 0)
            parameter_types = [qubit_type] * op.num_qubits + [param_type] * num_parameters

            module.add_external_function(pyqir_op, parameter_types=parameter_types)

        args = []
        if type_ is InstrType.MEASUREMENT:
            block = "measurements"

            for qb, b in zip(op.qubits, op.bits):
                _, qubit_idx = circuit.find_qubit(qb)
                _, bit_idx = circuit.find_bit(b)
                args = (module.qubits[qubit_idx], module.results[bit_idx])

                measurement_instruction = Instruction(InstrType.MEASUREMENT, pyqir_op, args)
                module.add_instruction("measurements", measurement_instruction)

                output_instruction = Instruction(InstrType.OUTPUT, "out", [args[1]])
                module.add_instruction("output", output_instruction)

        elif type_ is InstrType.BASE or (type_ is InstrType.EXTERNAL and add_external):
            if block == "measurements":
                raise ValueError(
                    "Mid-circuit measurements are not supported when compiling to QIR."
                )

            # if isinstance(op, ParametricOperation):
            if hasattr(op, "parameters"):
                args.extend(op.parameters)

            for qb in op.qubits:
                _, qubit_idx = circuit.find_qubit(qb)
                args.append(module.qubits[qubit_idx])

            if type_ is InstrType.EXTERNAL:
                body_instruction = module.get_external_instruction(pyqir_op, args)
            else:
                body_instruction = Instruction(InstrType.BASE, pyqir_op, args)

            module.add_instruction("body", body_instruction)
        elif type_ is InstrType.DECOMPOSE or (type_ is InstrType.EXTERNAL and not add_external):
            raise NotImplementedError("Operation decompositions not supported.")
        else:
            raise NotImplementedError(f"Support not implemented for {op.__class__.name} gate.")

    return module
