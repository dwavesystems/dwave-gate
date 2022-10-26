import inspect
import math
from random import random
from typing import Generator

import numpy as np
import pytest

import dwgms.operations as ops
from dwgms.circuit import Circuit
from dwgms.operations.base import (
    ControlledOperation,
    Operation,
    ParametricOperation,
    create_operation,
)
from dwgms.tools import build_unitary


def get_operations(op_type: str = None) -> Generator:
    """Returns a generator for the operations.

    The returned generetor yields the corresponding operations declared in ``__all__`` in the
    order that they are written. Only non-abstract classes inheriting directly or indirectly
    from ``Operation`` are yielded.

    Args:
        op_type: The requested classtype of the operations yielded. If ``None`` or different from
            any of the values listed below, all operations will be yielded.
            - parametric: only parametric operations
            - controlled: only controlled operations
            - other: all operations except the ones described above
    """
    type_map = {
        "parametric": ParametricOperation,
        "controlled": ControlledOperation,
    }

    for opstr in ops.__all__:
        op = getattr(ops, opstr)

        # don't yield abstract base classes
        if getattr(op, "__abstractmethods__", False):
            continue

        # return all operations except for those in "type_map"
        if op_type == "other":
            other = all(op not in type_.__subclasses__() for type_ in type_map.values())
            if inspect.isclass(op) and issubclass(op, Operation) and other:
                yield op

        elif inspect.isclass(op) and issubclass(op, type_map.get(op_type, Operation)):
            yield op


def z_op() -> Operation:
    """Helper function to create a custom Z operation, for testing the custom operations created by
    the ``create_operation`` function.

    Returns:
        Operation: Custom Z operation.
    """
    circuit = Circuit(1)

    with circuit.context as q:
        ops.Hadamard(q[0])
        ops.X(q[0])
        ops.Hadamard(q[0])

    return create_operation(circuit, label="CustomZ")


def rot_op() -> Operation:
    """Helper function to create a custom Z operation, for testing the custom operations created by
    the ``create_operation`` function.

    Returns:
        Operation: Custom rotation operation.
    """
    circuit = Circuit(1, parametric=True)

    with circuit.context as (p, q):
        ops.RZ(p[0], q[0])
        ops.RY(p[1], q[0])
        ops.RZ(p[2], q[0])

    return create_operation(circuit, label="CustomRot")


@pytest.mark.parametrize("Op", list(get_operations()) + [z_op(), rot_op()])
class TestOperations:
    """Unit tests that apply for all operations."""

    # NOTE: more of an integration test to assert that decompositions are correct;
    # also, decompositions are currently just operations applied to the same qubits,
    # using (if applicable) the same parameters
    def test_decompositions(self, Op, empty_circuit):
        """Test that decompositions are correct. Only tests operations with a ``_decomposition``
        attribute."""
        empty_circuit.add_qregister(Op.num_qubits)

        # check if operation has a decomposition defined
        if not getattr(Op, "_decomposition", None):
            pytest.skip(reason="No decomposition implemented.")

        decomposition = [getattr(ops, op) for op in Op.decomposition]
        qubits = [[f"q{i}" for i in range(op.num_qubits)] for op in decomposition]

        # if operation is parametric, then add parameters
        if issubclass(Op, ParametricOperation):
            # NOTE: uses random parameter values from 0 to 2pi
            params = [
                [random() * 2 * math.pi for _ in range(op.num_parameters)] for op in decomposition
            ]
            op_instances = [op(p, q) for op, p, q in zip(decomposition, params, qubits)]
            expected = Op([p for i in params for p in i]).matrix

        # if operation is controlled, use control/target instead of qubits
        elif issubclass(Op, ControlledOperation):
            op_instances = [
                op(q[: op.num_control], q[op.num_control :]) for op, q in zip(decomposition, qubits)
            ]
            expected = Op.matrix

        # else just use operation as is
        else:
            op_instances = [op(q) for op, q in zip(decomposition, qubits)]
            expected = Op.matrix

        # create the circuit and build the unitary
        empty_circuit.append(op_instances)
        unitary = build_unitary(empty_circuit)

        # compare built unitary with operation matrix representation
        assert np.allclose(unitary, expected)


class TestCustomOperations:
    """Unit tests for custom operation inheriting from one of the base classes."""

    def test_missing_num_qubits_attribute(self):
        """Test that the correct exception is raised when a subclass to ``Operation`` is missing the
        ``_num_qubits`` class attribute."""

        class CustomOp(Operation):
            pass

        with pytest.raises(AttributeError, match="missing class attributes '_num_qubits'"):
            CustomOp.num_qubits

    def test_missing_num_control_attribute(self):
        """Test that the correct exception is raised when a subclass to ``ControlledOperation`` is missing
        the ``_num_control`` class attribute."""

        class CustomOp(ControlledOperation):
            _num_target = 1

        with pytest.raises(AttributeError, match="missing class attributes '_num_control'"):
            CustomOp.num_qubits

    def test_missing_num_target_attribute(self):
        """Test that the correct exception is raised when a subclass to ``ControlledOperation`` is missing
        the ``_num_target`` class attribute."""

        class CustomOp(ControlledOperation):
            _num_control = 1

        with pytest.raises(AttributeError, match="missing class attributes '_num_control'"):
            CustomOp.num_qubits

    def test_num_parameters_attribute(self):
        """Test the ``num_parameters`` attribute."""

        class CustomOp(ParametricOperation):
            _num_params = 2

        assert CustomOp.num_parameters == 2

    def test_missing_num_parameters_attribute(self):
        """Test that the correct exception is raised when a subclass to ``ParametricOperation`` is missing
        the ``_num_params`` class attribute."""

        class CustomOp(ParametricOperation):
            pass

        with pytest.raises(AttributeError, match="missing class attribute '_num_params'"):
            CustomOp.num_parameters


@pytest.mark.parametrize("ParamOp", list(get_operations("parametric")) + [rot_op()])
class TestParametricOperations:
    """Unit tests for all parametric operations."""

    def test_initialize_operation(self, ParamOp):
        """Test initializing a parametric operation."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters)]
        qubits = tuple(f"q{i}" for i in range(ParamOp.num_qubits))
        label = ParamOp.label

        op = ParamOp(params, qubits)

        assert op.parameters == params
        assert op.qubits == qubits

        assert op.label == f"{label}({params})"

    def test_incorrect_number_of_params(self, ParamOp):
        """Test calling a parametric operation with the incorrect number of parameters."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters + 2)]

        with pytest.raises(
            ValueError,
            match=f"requires {ParamOp.num_parameters} parameters, got {ParamOp.num_parameters + 2}",
        ):
            ParamOp(params)

    def test_initializing_gate_in_context(self, empty_circuit, ParamOp):
        """Test initializing parametric operation within a context."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters)]

        empty_circuit.add_qregister(ParamOp.num_qubits)
        with empty_circuit.context as q:
            op = ParamOp(params, q)

        assert empty_circuit.circuit == [op]

    def test_initializing_gate_in_context_without_qubits(self, empty_circuit, ParamOp):
        """Test that the correct error is raised when initializing a parametric operation inside a
        context without declaring qubits."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters)]

        empty_circuit.add_qregister(ParamOp.num_qubits)
        with pytest.raises(TypeError, match="Qubits required when applying gate within context."):
            with empty_circuit.context:
                ParamOp(params)

    def test_initializing_gate_with_invalid_qubits(self, empty_circuit, ParamOp):
        """Test that the correct error is raised when initializing a parametric operation inside a
        context on non-existing qubits."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters)]

        empty_circuit.add_qregister(ParamOp.num_qubits + 9)
        with pytest.raises(
            ValueError, match=f"requires {ParamOp.num_qubits} qubits, got {ParamOp.num_qubits + 9}."
        ):
            with empty_circuit.context as q:
                ParamOp(params, q)

    def test_applying_operation_instance(self, empty_circuit, ParamOp):
        """Test applying an instance of a parametric operation within a context."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters)]

        empty_circuit.add_qregister(ParamOp.num_qubits)

        op = ParamOp(params, empty_circuit.qubits)
        with empty_circuit.context:
            op()

        assert empty_circuit.circuit == [op]

    def test_applying_operation_instance_deferred(self, empty_circuit, ParamOp):
        """Test applying an instance of a parametric operation within a context, but deferring qubit
        declaration till application."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters)]

        empty_circuit.add_qregister(ParamOp.num_qubits)

        op = ParamOp(params)
        with empty_circuit.context:
            op(empty_circuit.qubits)

        assert empty_circuit.circuit == [ParamOp(params, empty_circuit.qubits)]

    def test_set_qubits(self, ParamOp):
        """Test changing the qubits of a parametric operation instance."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters)]
        qubits = tuple(f"custom_qubit{i}" for i in range(ParamOp.num_qubits))

        op = ParamOp(params)

        assert op.qubits is None
        op.qubits = qubits
        assert op.qubits is qubits

    def test_warning_set_qubits(self, ParamOp):
        """Test that the correct warning is raised when changing already set qubits."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters)]
        qubits_0 = tuple(f"q{i}" for i in range(ParamOp.num_qubits))
        qubits_1 = tuple(f"custom_qubit{i}" for i in range(ParamOp.num_qubits))

        op = ParamOp(params, qubits_0)

        with pytest.warns(match="Changing qubits on which"):
            op.qubits = qubits_1

    def test_set_invalid_qubits(self, ParamOp):
        """Test that the correct error is raised when setting an invalid number of qubits."""
        params = [f"p{i}" for i in range(ParamOp.num_parameters)]
        # create too many qubits, num_qubits + 6
        qubits = tuple(f"custom_qubit{i}" for i in range(ParamOp.num_qubits + 6))

        op = ParamOp(params)

        assert op.qubits is None
        with pytest.raises(
            ValueError,
            match=f"Operation '{ParamOp.label}' requires {op.num_qubits} qubits, got {op.num_qubits + 6}.",
        ):
            op.qubits = qubits


@pytest.mark.parametrize("ControlledOp", list(get_operations("controlled")))
class TestControlledOperations:
    """Unit tests for all controlled operations."""

    def test_initialize_operation(self, ControlledOp):
        """Test initializing a controlled operation."""
        control = tuple(f"c{i}" for i in range(ControlledOp.num_control))
        target = tuple(f"t{i}" for i in range(ControlledOp.num_control))
        label = ControlledOp.label

        op = ControlledOp(control, target)

        assert op.control == control
        assert op.target == target
        assert op.label == label

    def test_initializing_gate_in_context(self, empty_circuit, ControlledOp):
        """Test initializing a controlled operation within a context."""
        empty_circuit.add_qregister(ControlledOp.num_qubits)
        with empty_circuit.context as q:
            op = ControlledOp(q[: ControlledOp.num_control], q[ControlledOp.num_control :])

        assert empty_circuit.circuit == [op]

    def test_initializing_gate_in_context_without_qubits(self, empty_circuit, ControlledOp):
        """Test that the correct error is raised when initializing a controlled operation inside a
        context without declaring qubits."""
        empty_circuit.add_qregister(ControlledOp.num_qubits)
        with pytest.raises(TypeError, match="Qubits required when applying gate within context."):
            with empty_circuit.context:
                ControlledOp()

    def test_initializing_gate_with_invalid_qubits(self, empty_circuit, ControlledOp):
        """Test that the correct error is raised when initializing a controlled operation inside a
        context on non-existing qubits."""
        empty_circuit.add_qregister(ControlledOp.num_qubits + 9)
        with pytest.raises(
            ValueError,
            match=f"requires {ControlledOp.num_qubits} qubits, got {ControlledOp.num_qubits + 9}.",
        ):
            with empty_circuit.context as q:
                ControlledOp(q[: ControlledOp.num_control], q[ControlledOp.num_control :])

    def test_applying_operation_instance(self, empty_circuit, ControlledOp):
        """Test applying an instance of a controlled operation within a context."""
        empty_circuit.add_qregister(ControlledOp.num_qubits)

        control = empty_circuit.qubits[: ControlledOp.num_control]
        target = empty_circuit.qubits[ControlledOp.num_control :]

        op = ControlledOp(control, target)
        with empty_circuit.context:
            op()

        assert empty_circuit.circuit == [op]

    def test_applying_operation_instance_deferred(self, empty_circuit, ControlledOp):
        """Test applying an instance of a controlled operation within a context, but deferring qubit
        declaration till application."""
        empty_circuit.add_qregister(ControlledOp.num_qubits)

        control = empty_circuit.qubits[: ControlledOp.num_control]
        target = empty_circuit.qubits[ControlledOp.num_control :]

        op = ControlledOp()
        with empty_circuit.context:
            op(control, target)

        assert empty_circuit.circuit == [ControlledOp(control, target)]


@pytest.mark.parametrize("Op", list(get_operations("other")) + [z_op()])
class TestOtherOperations:
    """Unit tests for all other operations."""

    def test_initialize_operation(self, Op):
        """Test initializing an operation."""
        qubits = tuple(f"c{i}" for i in range(Op.num_qubits))
        label = Op.label

        op = Op(qubits)

        assert op.qubits == qubits
        assert op.label == label

    def test_initializing_gate_in_context(self, empty_circuit, Op):
        """Test initializing an operation within a context."""
        empty_circuit.add_qregister(Op.num_qubits)
        with empty_circuit.context as q:
            op = Op(q)

        assert empty_circuit.circuit == [op]

    def test_initializing_gate_in_context_without_qubits(self, empty_circuit, Op):
        """Test that the correct error is raised when initializing an operation inside a
        context without declaring qubits."""
        empty_circuit.add_qregister(Op.num_qubits)
        with pytest.raises(TypeError, match="Qubits required when applying gate within context."):
            with empty_circuit.context:
                Op()

    def test_initializing_gate_with_invalid_qubits(self, empty_circuit, Op):
        """Test that the correct error is raised when initializing an operation inside a
        context on non-existing qubits."""
        empty_circuit.add_qregister(Op.num_qubits + 9)
        with pytest.raises(
            ValueError, match=f"requires {Op.num_qubits} qubits, got {Op.num_qubits + 9}."
        ):
            with empty_circuit.context as q:
                Op(q)

    def test_applying_operation_instance(self, empty_circuit, Op):
        """Test applying an instance of an operation within a context."""
        empty_circuit.add_qregister(Op.num_qubits)

        op = Op(empty_circuit.qubits)
        with empty_circuit.context:
            op()

        assert empty_circuit.circuit == [op]

    def test_applying_operation_instance_deferred(self, empty_circuit, Op):
        """Test applying an instance of an operation within a context, but deferring qubit
        declaration till application."""
        empty_circuit.add_qregister(Op.num_qubits)

        op = Op()
        with empty_circuit.context:
            op(empty_circuit.qubits)

        assert empty_circuit.circuit == [Op(empty_circuit.qubits)]

    def test_set_qubits(self, Op):
        """Test changing the qubits of an operation instance."""
        qubits = tuple(f"custom_qubit{i}" for i in range(Op.num_qubits))

        op = Op()

        assert op.qubits is None
        op.qubits = qubits
        assert op.qubits is qubits

    def test_warning_set_qubits(self, Op):
        """Test that the correct warning is raised when changing already set qubits."""
        qubits_0 = tuple(f"q{i}" for i in range(Op.num_qubits))
        qubits_1 = tuple(f"custom_qubit{i}" for i in range(Op.num_qubits))

        op = Op(qubits_0)

        with pytest.warns(match="Changing qubits on which"):
            op.qubits = qubits_1

    def test_set_invalid_qubits(self, Op):
        """Test that the correct error is raised when setting an invalid number of qubits."""
        # create too many qubits, num_qubits + 6
        qubits = tuple(f"custom_qubit{i}" for i in range(Op.num_qubits + 6))

        op = Op()

        assert op.qubits is None
        with pytest.raises(
            ValueError,
            match=f"Operation '{Op.label}' requires {op.num_qubits} qubits, got {op.num_qubits + 6}.",
        ):
            op.qubits = qubits
