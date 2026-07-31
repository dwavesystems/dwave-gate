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

"""Implementations of various simple algorithms in QCDL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .components import QcdlModule, Scope, procedure
from .constants import LogicalOutcomeToInteger

if TYPE_CHECKING:
    from .registers import Register


def _get_receivers(
    sender: QcdlModule,
    register: Register,
    receivers: list[QcdlModule] | None = None,
) -> list[QcdlModule]:
    if receivers is None:
        receivers = [q for q in register.qcdl_modules if repr(q) != repr(sender)]
    else:
        if sender in receivers:
            raise ValueError(f"{sender=} can't also be a receiver")
    return receivers


@procedure
def mirror_bool_register(
    sender: QcdlModule,
    register: Register,
    receivers: list[QcdlModule] | None = None,
) -> None:
    """Mirror a Boolean register to registers associated with other qubits.

    May be useful for a :class:`~dwave.gate.qcdl.registers.Register` object used
    for the outcome of a :func:`~dwave.gate.qcdl.operations.mced` operation. See
    the :ref:`qcdl_advanced_registers_mirroring` section to learn about
    mirroring.

    This function requires the following:

    *   Allocate the :class:`~dwave.gate.qcdl.registers.Register` to all the
        relevant qubits.
    *   A value of either :math:`0` or :math:`1` in the register associated with
        the qubit specified by the ``sender`` parameter.

    Args:
        sender: Qubit associated with the register you are assigning the
            measurement outcome to.
        register: :class:`~dwave.gate.qcdl.registers.Register` object to mirror.
        receivers: Qubits that need their associated registers updated. If not
            specified, defaults to the other qubits associated with the register
            selected by the ``Register`` parameter.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl, Scope
            from dwave.gate.qcdl.implementations import mirror_bool_register
            from dwave.gate.qcdl.operations import h, mced, x

            @qcdl(3)
            def mirror_bool_example(q0, q1, q2):
                sc = Scope(q0, q1, q2)
                h(q0)
                r0 = Register(sc.qcdl_modules, name="r0")
                receivers = [qubit for qubit in sc.qcdl_modules if qubit != q0]
                mced(q0, register=r0)
                mirror_bool_register(sender=q0, register=r0, receivers=receivers)

            qcdl_dict = mirror_bool_example()
    """
    receivers = _get_receivers(sender=sender, register=register, receivers=receivers)
    if not receivers:
        return
    receivers_scope = Scope(*receivers, use_scope_id=False)
    dst_reg = receivers_scope.Register(name=register.name, alias=True)

    sender.comment()
    sender.comment("broadcasting bool: register == 1")
    sender.one_to_all(receivers_scope, register == 1)
    with receivers_scope.If(None) as _Else:
        dst_reg <<= 1
    with _Else():
        dst_reg <<= 0


@procedure
def mirror_measurement_register(
    sender: QcdlModule,
    register: Register,
    receivers: list[QcdlModule] | None = None,
) -> None:
    """Mirror a two-bit qubit measurement to registers associated with other
    qubits.

    May be useful for a :class:`~dwave.gate.qcdl.registers.Register` object used
    for the outcome of a :func:`~dwave.gate.qcdl.operations.measurement`
    operation. See the :ref:`qcdl_advanced_registers_mirroring` section to learn
    about mirroring.

    This function requires the following:

    *   Allocate the :class:`~dwave.gate.qcdl.registers.Register` to all the
        relevant qubits.
    *   The measurement outcome is written to the register associated with the
        qubit specified by the ``sender`` parameter.
    *   The measurement outcome is two bits (a dual-rail qubit measurement).

    Args:
        sender: Qubit associated with the register you are assigning the
            measurement outcome to.
        register: :class:`~dwave.gate.qcdl.registers.Register` object to mirror.
        receivers: Qubits that need their associated registers updated. If not
            specified, defaults to the other qubits associated with the register
            selected by the ``Register`` parameter.

    Examples:

        .. testcode::

            from dwave.gate.qcdl import qcdl, Scope
            from dwave.gate.qcdl.implementations import mirror_measurement_register
            from dwave.gate.qcdl.operations import h, measure, x

            @qcdl(3)
            def mirror_measurement_example(q0, q1, q2):
                sc = Scope(q0, q1, q2)
                h(q0)
                r0 = Register(sc.qcdl_modules, name="r0")
                receivers = [qubit for qubit in sc.qcdl_modules if qubit != q0]
                measure(q0, register=r0)
                mirror_measurement_register(sender=q0, register=r0, receivers=receivers)

            qcdl_dict = mirror_measurement_example()
    """
    receivers = _get_receivers(sender=sender, register=register, receivers=receivers)
    if not receivers:
        return
    receivers_scope = Scope(*receivers, use_scope_id=False)
    dst_reg = receivers_scope.Register(name=register.name, alias=True)
    dst_reg <<= LogicalOutcomeToInteger.ZERO.value

    for val in [LogicalOutcomeToInteger.ONE, LogicalOutcomeToInteger.SPLAT]:
        sender.comment()
        sender.comment(f"broadcasting bit: register == {val=}")
        sender.one_to_all(receivers_scope, register == val.value)
        with receivers_scope.If(None):
            dst_reg <<= val.value
