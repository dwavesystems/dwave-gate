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

"""API for gates and measurements.

Because the transpiler used by both the compiler and the simulator is
implemented using `Qiskit <https://github.com/Qiskit/qiskit>`_, all
`QuantumCircuit <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.QuantumCircuit>`_
gates are supported.

The examples below illustrate operations.

Examples:
    This example uses the :mod:`~dwave.gate.qcdl.operations` API to add gates and
    measurements to a procedure.

    .. testcode::

        from dwave.gate.qcdl import qcdl
        from dwave.gate.qcdl.operations import cx, h, measure

        @qcdl()
        def operations_api(q0, q1):
            h(q0)
            cx(q0, q1)
            measure(q0)
            measure(q1)

        qcdl_dict = operations_api()

    This example is equivalent to the above example.

    .. testcode::

        from dwave.gate.qcdl import qcdl

        @qcdl()
        def qcdl_module_methods(q0, q1):
            q0.h()
            q0.cx(q1)
            q0.measure()
            q1.measure()

        qcdl_dict = qcdl_module_methods()
"""

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np

from . import implementations
from .components import QcdlModule
from .exceptions import QCDLUserError
from .registers import FixedPointRegister, Register

AngleType: TypeAlias = float | FixedPointRegister

# Operations


def initialize(*qubits: QcdlModule) -> None:
    """Initialize qubits.

    This function has non-deterministic duration because the implementation is
    to loop until all qubits are reset. This means that currently all qubits in
    the program must be included in the call.

    .. note:: This function is implicitly added by default at the beginning of
        all programs.

    Args:
        *qubits (QcdlModule): Qubits to initialize. It is not an error to do
            so, but unused qubits should not be included.

    """
    qubits[0].initialize(*qubits[1:])


def barrier(*qubits: QcdlModule, label: str | None = None) -> None:
    """Place a barrier on qubits.

    The transpiler does not combine gates across a barrier. This is a directive
    to the transpiler and for dynamical decoupling, and is otherwise handled
    like a comment.

    .. note:: This operation is different from a
        :meth:`~dwave.gate.qcdl.QcdlModuleContainer.sync` method, which is used
        to control the order that the compiler schedules operations.

    Args:
        *qubits (QcdlModule): The qubits to put a barrier on.
        label (str, optional): An annotation.

    Examples:

        .. testcode::
            :skipif: True   # TODO: figure out why this test fails

            from dwave.gate.qcdl import print_qcdl, qcdl
            from dwave.gate.qcdl.operations import barrier, measure, x

            @qcdl()
            def use_barrier(q0):
                x(q0)
                barrier(q0, label="Separate two X gates")
                x(q0)
                measure(q0)

            qcdl_dict = use_barrier()
            print_qcdl(qcdl_dict)

        The code above prints the following QCDL.

        .. testoutput::
            :skipif: True   # TODO: figure out why this test fails
            :options: +NORMALIZE_WHITESPACE

            begin quantum
                x(q0)
                q0.barrier(label="Separate two X gates")
                x(q0)
                measure(q0, log=True)
            end quantum
    """
    kwargs = {}
    if label is not None:
        kwargs["label"] = label
    qubits[0].barrier(*qubits[1:], **kwargs)


def measure(
    qubit: QcdlModule,
    log: bool = True,
    tag: str | None = None,
    register: Register | None = None,
    mirror: bool = True,
) -> None:
    """Measure a qubit.

    Args:
        qubit (QcdlModule): The qubit to measure.
        log (bool, optional): If True, the measurement result is logged, meaning
            it is included in the returned array for this qubit.
        tag (str | None, optional): Name for this measurement, used to organize the
            results. Tagging is not compatible with programs that do not
            generate the same number of measurements per shot, which requires
            real-time measurements.
        register (Register | None, optional): Register for storing the outcome.
            Supported values are :math:`0`, :math:`1`, or :math:`-1` (for
            splat). (See the :class:`~dwave.gate.qcdl.LogicalOutcomeToInteger`
            class.)
        mirror (bool): If True and if using a register, mirrors the outcome to
            all qubits in the register. If False, the register is updated only
            for the selected qubit. Mirroring may be skipped or deferred because
            it can be expensive.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure

            @qcdl(1)
            def measurement(q0):
                r0 = q0.Register(name="r0")
                h(q0)
                measure(q0, tag="my measurement", register=r0)

            qcdl_dict: dict = measurement()
    """
    kwargs: dict[str, Any] = dict(log=log)
    if register:
        kwargs["register"] = register
    if tag is not None:
        if not isinstance(tag, str):
            raise TypeError(
                f"tag must be a str, {tag=} of type {type(tag)} is not allowed"
            )
        kwargs["tag"] = tag
    qubit.procedure.add_statement(None, "measure", [qubit], kwargs)

    if register and mirror and len(register.qcdl_modules) > 1:
        implementations.mirror_measurement_register(sender=qubit, register=register)


def mced(qubit: QcdlModule, register: Register, mirror: bool = True) -> None:
    """Perform a non-destructive mid-circuit erasure detection (MCED).

    Args:
        qubit (QcdlModule): The qubit to inspect.
        register (Register): Register used for storing the outcome. Supported
            outcomes are:

            *   0: No erasure detected
            *   1: Erasure detected.
        mirror (bool): If True and if using a register, mirrors the outcome to
            all qubits in the register. If False, the register is updated only
            for the selected qubit. Mirroring may be skipped or deferred because
            it can be expensive.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, mced, measure, rx

            @qcdl(1)
            def mced_use(q0):
                r0 = q0.Register()
                h(q0)
                mced(q0, register=r0)
                rx(q0, phi=0.1)
                measure(q0)

            qcdl_dict: dict = mced_use()
    """
    qubit.procedure.add_statement(None, "mced", [qubit], dict(register=register))
    if register and mirror and len(register.qcdl_modules) > 1:
        implementations.mirror_bool_register(sender=qubit, register=register)


# 1 Qubit Non-Parameterized Gates


def x(qubit: QcdlModule) -> None:
    """`X <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.XGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, x

            @qcdl(1)
            def x_gate(q0):
                x(q0)
                measure(q0)

            qcdl_dict: dict = x_gate()
    """
    qubit.procedure.add_statement(None, "x", [qubit], None)


def sx(qubit: QcdlModule) -> None:
    r"""`Square-root of X <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.SXGate>`_
    (:math:`\sqrt X`) gate.

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, sx

            @qcdl(1)
            def sx_gate(q0):
                sx(q0)
                measure(q0)

            qcdl_dict: dict = sx_gate()
    """
    qubit.procedure.add_statement(None, "sx", [qubit], None)


def y(qubit: QcdlModule) -> None:
    """`Y <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.YGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, y

            @qcdl(1)
            def y_gate(q0):
                y(q0)
                measure(q0)

            qcdl_dict: dict = y_gate()
    """
    qubit.procedure.add_statement(None, "y", [qubit], None)


def sy(qubit: QcdlModule) -> None:
    r"""SQRT of Y gate.
    TODO it's not a qiskit gate

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, sy

            @qcdl(1)
            def sy_gate(q0):
                sy(q0)
                measure(q0)

            qcdl_dict: dict = sy_gate()
    """
    qubit.procedure.add_statement(None, "ry", [qubit, np.pi / 2], None)


def sydg(qubit: QcdlModule) -> None:
    r"""SQRT of Y_adjoint gate.
    TODO it's not a qiskit gate

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, sydg

            @qcdl(1)
            def sydg_gate(q0):
                sydg(q0)
                measure(q0)

            qcdl_dict: dict = sydg_gate()
    """
    qubit.procedure.add_statement(None, "ry", [qubit, -np.pi / 2], None)


def z(qubit: QcdlModule) -> None:
    """`Z <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.ZGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, z

            @qcdl(1)
            def z_gate(q0):
                z(q0)
                measure(q0)

            qcdl_dict: dict = z_gate()
    """
    qubit.procedure.add_statement(None, "z", [qubit], None)


def s(qubit: QcdlModule) -> None:
    """`S <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.SGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, s

            @qcdl(1)
            def s_gate(q0):
                s(q0)
                measure(q0)

            qcdl_dict: dict = s_gate()
    """
    qubit.procedure.add_statement(None, "s", [qubit], None)


def sdg(qubit: QcdlModule) -> None:
    r"""`S-adjoint <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.SdgGate>`_
    (:math:`S^\dagger`) gate.

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, sdg

            @qcdl(1)
            def sdg_gate(q0):
                sdg(q0)
                measure(q0)

            qcdl_dict: dict = sdg_gate()

    """
    qubit.procedure.add_statement(None, "sdg", [qubit], None)


def t(qubit: QcdlModule) -> None:
    r"""`T <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.TGate>`_
    (:math:`\sqrt[4]{Z}`) gate.

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, t

            @qcdl(1)
            def t_gate(q0):
                t(q0)
                measure(q0)

            qcdl_dict: dict = t_gate()
    """
    qubit.procedure.add_statement(None, "t", [qubit], None)


def tdg(qubit: QcdlModule) -> None:
    r"""`T-adjoint <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.TdgGate>`_
    (:math:`T^\dagger`) gate.

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, tdg

            @qcdl(1)
            def tdg_gate(q0):
                tdg(q0)
                measure(q0)

            qcdl_dict: dict = tdg_gate()
    """
    qubit.procedure.add_statement(None, "tdg", [qubit], None)


def h(qubit: QcdlModule) -> None:
    """`Hadamard <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.HGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure

            @qcdl(1)
            def h_gate(q0):
                h(q0)
                measure(q0)

            qcdl_dict: dict = h_gate()
    """
    qubit.procedure.add_statement(None, "h", [qubit], None)


# 1 Qubit Parameterized Gates


def rx(qubit: QcdlModule, phi: AngleType) -> None:
    r"""Single-qubit
    `X-axis rotation <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RXGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.
        phi: :math:`\phi` angle of rotation about the X axis.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure, rx

            @qcdl(1)
            def rx_gate(q0):
                h(q0)
                rx(q0, phi=0.1)
                measure(q0)

            qcdl_dict: dict = rx_gate()
    """
    qubit.procedure.add_statement(None, "rx", [qubit, phi], None)


def ry(qubit: QcdlModule, phi: AngleType) -> None:
    r"""Single-qubit
    `Y-axis rotation <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RYGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.
        phi: :math:`\phi` angle of rotation about the Y axis.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure, ry

            @qcdl(1)
            def ry_gate(q0):
                h(q0)
                ry(q0, phi=0.1)
                measure(q0)

            qcdl_dict: dict = ry_gate()
    """
    qubit.procedure.add_statement(None, "ry", [qubit, phi], None)


def rz(qubit: QcdlModule, phi: AngleType) -> None:
    r"""Single-qubit
    `Z-axis rotation <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RZGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.
        phi: :math:`\phi` angle of rotation about the Z axis.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure, rz

            @qcdl(1)
            def rz_gate(q0):
                h(q0)
                rz(q0, phi=0.1)
                measure(q0)

            qcdl_dict: dict = rz_gate()
    """
    qubit.procedure.add_statement(None, "rz", [qubit, phi], None)


def p(qubit: QcdlModule, theta: AngleType) -> None:
    r"""`Phase <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.PhaseGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.
        phi: :math:`\theta` angle of rotation about the Z axis.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure, p

            @qcdl(1)
            def p_gate(q0):
                h(q0)
                p(q0, theta=0.1)
                measure(q0)

            qcdl_dict: dict = p_gate()
    """
    qubit.procedure.add_statement(None, "p", [qubit, theta], None)


def u(qubit: QcdlModule, theta: AngleType, phi: AngleType, lam: AngleType) -> None:
    r"""Single-qubit
    `generic U <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.UGate>`_
    gate.

    Args:
        qubit: Qubit on which to apply the gate.
        theta: :math:`\theta` angle of rotation.
        phi: :math:`\phi` angle of rotation.
        lam: :math:`\lambda` angle of rotation.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import measure, u

            @qcdl(1)
            def u_gate(q0):
                u(q0, theta=0.1, phi=0.2, lam=0.3)
                measure(q0)

            qcdl_dict: dict = u_gate()
    """
    qubit.procedure.add_statement(None, "u", [qubit, theta, phi, lam], None)


# 2 Qubit Non-Parameterized Gates


def swap(qubit1: QcdlModule, qubit2: QcdlModule) -> None:
    """`Swap <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.SwapGate>`_
    gate.

    Swaps quantum states between ``qubit1`` and ``qubit2``.

    Args:
        qubit1: A qubit.
        qubit2: A qubit.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure, swap

            @qcdl(2)
            def swap_gate(q0, q1):
                h(q0)
                swap(q0, q1)
                measure(q1)

            qcdl_dict: dict = swap_gate()
    """
    qubit1.procedure.add_statement(None, "swap", [qubit1, qubit2], None)


def cx(control_qubit: QcdlModule, target_qubit: QcdlModule) -> None:
    """`Controlled-X <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CXGate>`_
    gate.

    Args:
        control_qubit: Control qubit.
        target_qubit: Targeted qubit.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import cx, h, measure

            @qcdl(2)
            def cx_gate(q0, q1):
                h(q0)
                cx(control_qubit=q0, target_qubit=q1)
                measure(q1)

            qcdl_dict: dict = cx_gate()
    """
    control_qubit.procedure.add_statement(
        None, "cx", [control_qubit, target_qubit], None
    )


def cy(control_qubit: QcdlModule, target_qubit: QcdlModule) -> None:
    """`Controlled-Y <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CYGate>`_
    gate.

    Args:
        control_qubit: Control qubit.
        target_qubit: Targeted qubit.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import cy, h, measure

            @qcdl(2)
            def cy_gate(q0, q1):
                h(q0)
                cy(control_qubit=q0, target_qubit=q1)
                measure(q1)

            qcdl_dict: dict = cy_gate()
    """
    control_qubit.procedure.add_statement(
        None, "cy", [control_qubit, target_qubit], None
    )


def cz(control_qubit: QcdlModule, target_qubit: QcdlModule) -> None:
    """`Controlled-Z <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CZGate>`_
    gate.

    Args:
        control_qubit: Control qubit.
        target_qubit: Targeted qubit.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import cz, h, measure

            @qcdl(2)
            def cz_gate(q0, q1):
                h(q0)
                cz(control_qubit=q0, target_qubit=q1)
                measure(q1)

            qcdl_dict: dict = cz_gate()
    """
    control_qubit.procedure.add_statement(
        None, "cz", [control_qubit, target_qubit], None
    )


# 2 Qubit Parameterized Gates


def crx(control_qubit: QcdlModule, target_qubit: QcdlModule, theta: AngleType) -> None:
    r"""`Controlled-RX <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CRXGate>`_
    gate.

    Args:
        control_qubit: Control qubit.
        target_qubit: Targeted qubit.
        theta: :math:`\theta` angle of the rotation.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import crx, h, measure

            @qcdl(2)
            def crx_gate(q0, q1):
                h(q0)
                crx(control_qubit=q0, target_qubit=q1, theta=0.1)
                measure(q1)

            qcdl_dict: dict = crx_gate()
    """

    control_qubit.procedure.add_statement(
        None, "crx", [control_qubit, target_qubit, theta], None
    )


def cry(control_qubit: QcdlModule, target_qubit: QcdlModule, theta: AngleType) -> None:
    r"""`Controlled-RY <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CRYGate>`_
    gate.

    Args:
        control_qubit: Control qubit.
        target_qubit: Targeted qubit.
        theta: :math:`\theta` angle of the rotation.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import cry, h, measure

            @qcdl(2)
            def cry_gate(q0, q1):
                h(q0)
                cry(control_qubit=q0, target_qubit=q1, theta=0.1)
                measure(q1)

            qcdl_dict: dict = cry_gate()
    """

    control_qubit.procedure.add_statement(
        None, "cry", [control_qubit, target_qubit, theta], None
    )


def crz(control_qubit: QcdlModule, target_qubit: QcdlModule, theta: AngleType) -> None:
    r"""`Controlled-RZ <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CRZGate>`_
    gate.

    Args:
        control_qubit: Control qubit.
        target_qubit: Targeted qubit.
        theta: :math:`\theta` angle of the rotation.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import crz, h, measure

            @qcdl(2)
            def crz_gate(q0, q1):
                h(q0)
                crz(control_qubit=q0, target_qubit=q1, theta=0.1)
                measure(q1)

            qcdl_dict: dict = crz_gate()
    """

    control_qubit.procedure.add_statement(
        None, "crz", [control_qubit, target_qubit, theta], None
    )


def cp(control_qubit: QcdlModule, target_qubit: QcdlModule, theta: AngleType) -> None:
    r"""`Controlled-Phase <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CPhaseGate>`_
    gate.

    Args:
        control_qubit: Control qubit.
        target_qubit: Targeted qubit.
        theta: :math:`\theta` angle of the rotation.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import cp, h, measure

            @qcdl(2)
            def cp_gate(q0, q1):
                h(q0)
                cp(control_qubit=q0, target_qubit=q1, theta=0.1)
                measure(q1)

            qcdl_dict: dict = cp_gate()

    """
    control_qubit.procedure.add_statement(
        None, "cp", [control_qubit, target_qubit, theta], None
    )


def cu(
    control_qubit: QcdlModule,
    target_qubit: QcdlModule,
    theta: AngleType,
    phi: AngleType,
    lam: AngleType,
    gamma: AngleType,
) -> None:
    r"""`Controlled-U <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CUGate>`_
    gate.

    Args:
        control_qubit: Control qubit.
        target_qubit: Targeted qubit.
        theta: :math:`\theta` angle of rotation.
        phi: :math:`\phi` angle of rotation.
        lam: :math:`\lambda` angle of rotation.
        gamma: Global phase of the gate, if applicable.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import cu, h, measure

            @qcdl(2)
            def cu_gate(q0, q1):
                h(q0)
                cu(
                    control_qubit=q0, target_qubit=q1,
                    theta=0.1, phi=0.2, lam=0.3, gamma=-0.1
                )
                measure(q1)

            qcdl_dict: dict = cu_gate()
    """
    control_qubit.procedure.add_statement(
        None, "cu", [control_qubit, target_qubit, theta, phi, lam, gamma], None
    )


def rxx(qubit1: QcdlModule, qubit2: QcdlModule, theta: AngleType) -> None:
    r"""Two-qubit
    `XX-axis rotation <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RXXGate>`_
    gate.

    Args:
        qubit1: A qubit on which to apply the gate.
        qubit2: A qubit on which to apply the gate.
        theta: :math:`\theta` angle of rotation about the XX axis.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure, rxx

            @qcdl(2)
            def rxx_gate(q0, q1):
                h(q0)
                rxx(q0, q1, theta=0.1)
                measure(q0)

            qcdl_dict: dict = rxx_gate()
    """
    qubit1.procedure.add_statement(None, "rxx", [qubit1, qubit2, theta], None)


def ryy(qubit1: QcdlModule, qubit2: QcdlModule, theta: AngleType) -> None:
    r"""Two-qubit
    `YY-axis rotation <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RYYGate>`_
    gate.

    Args:
        qubit1: A qubit on which to apply the gate.
        qubit2: A qubit on which to apply the gate.
        theta: :math:`\theta` angle of rotation about the YY axis.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure, ryy

            @qcdl(2)
            def ryy_gate(q0, q1):
                h(q0)
                ryy(q0, q1, theta=0.1)
                measure(q0)

            qcdl_dict: dict = ryy_gate()
    """
    qubit1.procedure.add_statement(None, "ryy", [qubit1, qubit2, theta], None)


def rzz(qubit1: QcdlModule, qubit2: QcdlModule, theta: AngleType) -> None:
    r"""Two-qubit
    `ZZ-axis rotation <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RZZGate>`_
    gate.

    Args:
        qubit1: A qubit on which to apply the gate.
        qubit2: A qubit on which to apply the gate.
        theta: :math:`\theta` angle of rotation about the ZZ axis.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl
            from dwave.gate.qcdl.operations import h, measure, rzz

            @qcdl(2)
            def rzz_gate(q0, q1):
                h(q0)
                rzz(q0, q1, theta=0.1)
                measure(q0)

            qcdl_dict: dict = rzz_gate()
    """
    qubit1.procedure.add_statement(None, "rzz", [qubit1, qubit2, theta], None)
