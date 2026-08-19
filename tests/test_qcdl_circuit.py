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


"""Unit tests for the @qcdl decorator in qcdl_circuit.py.

Covers qubit-source modes (infer from signature, num_qubits, environment),
machine integration (get_system / set_up_systems / clean_up_systems),
f_keywords-based filtering of passed kwargs, and user-kwargs merging.
"""

import json
import pickle

import pytest

from dwave.gate.qcdl import QCDLUserError, qcdl
from dwave.gate.qcdl.qcdl_circuit import QCDLCircuit, _get_fspec
from dwave.gate.qcdl.qcdl_models import QCDLProgram

# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------


def _check_serializable(data):
    # ensure that qcdls are pickleable
    assert pickle.loads(pickle.dumps(data)) == data
    raw = data.model_dump(exclude_unset=True)
    assert json.loads(json.dumps(raw)) == raw


def _qcdl_decorator_reference(q0, q1, q2, my_kwarg):
    q0.comment(str(my_kwarg))
    for q in [q0, q1, q2]:
        q.h()


def _assert_statement_count(model, expected=4):
    assert len(model.program.statements) == expected


class FakeModule:
    """Minimal stand-in for an environment Module."""

    def __init__(self, name):
        self.name = name


class FakeEnv:
    """Minimal stand-in for an environment Environment."""

    def __init__(self, qubit_names):
        self._modules = [FakeModule(n) for n in qubit_names]

    def get_modules(self, include_couplers=True):
        return self._modules


class CallTracker:
    """Records which machine hooks were called and what they received."""

    def __init__(self):
        self.get_system_calls: list = []
        self.setup_called: bool = False
        self.cleanup_called: bool = False
        # Shallow snapshots of merged_kwargs taken at entry to each hook.
        self.setup_kwargs_snapshot: dict = {}
        self.cleanup_kwargs_snapshot: dict = {}


def make_fake_machine(env, tracker: CallTracker | None = None):
    """Return a FakeMachine class and its associated CallTracker.

    The machine promotes qubit/coupler names to QCDLModule objects via
    proc.q(name) inside set_up_systems, matching real machine behaviour.
    Non-qubit kwargs (e.g. user-supplied floats) are left untouched.
    """
    if tracker is None:
        tracker = CallTracker()

    class FakeMachine:
        environment = env

        @staticmethod
        def get_system(name):
            tracker.get_system_calls.append(name)
            return name

        @staticmethod
        def set_up_systems(systems, proc):
            tracker.setup_called = True
            tracker.setup_kwargs_snapshot = dict(systems)
            for q in list(systems):
                systems[q] = proc.q(q)

        @staticmethod
        def clean_up_systems(systems):
            tracker.cleanup_called = True
            tracker.cleanup_kwargs_snapshot = dict(systems)

    return FakeMachine, tracker


# ---------------------------------------------------------------------------
# 1. Qubit-source modes
# ---------------------------------------------------------------------------


class TestQCDLQubitSourceModes:
    """@qcdl infers/creates qubits differently depending on its arguments."""

    def test_infer_qubit_names_from_signature(self):
        """With no num_qubits or environment, qN/cN args are auto-injected."""
        received = {}

        @qcdl()
        def main(q0, q1):
            received["q0"] = q0
            received["q1"] = q1
            q0.x()
            q0.measure()
            q1.measure()

        result = main()
        assert isinstance(result, QCDLProgram)
        assert set(received) == {"q0", "q1"}

    def test_non_qubit_args_not_auto_injected(self):
        """Non-qubit arg names (e.g. 'angle') are not auto-created as qubits."""
        received = {}

        @qcdl()
        def main(q0, angle=None):
            received["angle"] = angle
            q0.x()
            q0.measure()

        result = main(angle=1.5)
        assert isinstance(result, QCDLProgram)
        assert received["angle"] == 1.5

    def test_coupler_name_inferred_from_signature(self):
        """cN arg names satisfy is_qubit_or_coupler_name and are injected."""
        received = {}

        @qcdl()
        def main(q0, c0):
            received["c0"] = c0
            q0.measure()

        result = main()
        assert isinstance(result, QCDLProgram)
        assert "c0" in received

    def test_num_qubits_creates_exact_count(self):
        """@qcdl(N) generates q0 … q(N-1)."""
        received = {}

        @qcdl(3)
        def main(q0, q1, q2):
            received.update({"q0": q0, "q1": q1, "q2": q2})
            q0.x()
            q0.measure()

        result = main()
        assert isinstance(result, QCDLProgram)
        assert set(received) == {"q0", "q1", "q2"}

    def test_environment_mode_uses_initialize_modules(self):
        """@qcdl(environment=env) initialises qubits from the environment."""
        env = FakeEnv(["q0", "q1"])
        received = {}

        @qcdl(environment=env)
        def main(q0, q1):
            received.update({"q0": q0, "q1": q1})
            q0.x()
            q0.measure()

        result = main()
        assert isinstance(result, QCDLProgram)
        assert set(received) == {"q0", "q1"}

    def test_environment_exposes_all_modules_to_varkw_function(self):
        """When f accepts **kwargs, all environment modules are forwarded."""
        env = FakeEnv(["q0", "q1", "q2"])
        received = {}

        @qcdl(environment=env)
        def main(**kwargs):
            received.update(kwargs)
            kwargs["q0"].x()
            kwargs["q0"].measure()

        result = main()
        assert isinstance(result, QCDLProgram)
        assert set(received) == {"q0", "q1", "q2"}

    def test_to_qcdlv2_returns_string(self):
        """to_qcdlv2=True returns the v2 textual format instead of a dict."""

        @qcdl(1, to_qcdlv2=True)
        def main(q0):
            q0.x()
            q0.measure()

        result = main()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 2. Machine integration
# ---------------------------------------------------------------------------


class TestQCDLMachine:
    """Tests for the machine= path through @qcdl."""

    def setup_method(self):
        self.env = FakeEnv(["q0", "q1"])

    def test_machine_and_environment_raises(self):
        """Supplying both machine= and environment= must raise QCDLUserError."""
        machine, _ = make_fake_machine(self.env)

        @qcdl(machine=machine, environment=self.env)
        def main(q0, q1):
            q0.x()
            q0.measure()

        with pytest.raises(QCDLUserError):
            main()

    def test_machine_get_system_called_for_each_module(self):
        """get_system is invoked once per key in module_kwargs."""
        machine, tracker = make_fake_machine(self.env)

        @qcdl(machine=machine)
        def main(q0, q1):
            q0.x()
            q0.measure()

        main()
        assert set(tracker.get_system_calls) == {"q0", "q1"}

    def test_machine_set_up_systems_called_before_body(self):
        """set_up_systems must complete before the decorated function body runs."""
        machine, tracker = make_fake_machine(self.env)

        @qcdl(machine=machine)
        def main(q0, q1):
            assert tracker.setup_called, "set_up_systems must run before f"
            q0.x()
            q0.measure()

        main()
        assert tracker.setup_called

    def test_machine_clean_up_systems_called_on_success(self):
        """clean_up_systems is called after a successful circuit body."""
        machine, tracker = make_fake_machine(self.env)

        @qcdl(machine=machine)
        def main(q0, q1):
            q0.x()
            q0.measure()

        main()
        assert tracker.cleanup_called

    def test_machine_clean_up_systems_called_on_exception(self):
        """clean_up_systems must run even when the circuit body raises."""
        machine, tracker = make_fake_machine(self.env)

        @qcdl(machine=machine)
        def main(q0, q1):
            raise RuntimeError("deliberate error")

        with pytest.raises(RuntimeError):
            main()

        assert tracker.cleanup_called

    def test_machine_set_up_receives_user_kwargs_in_merged(self):
        """set_up_systems is called with merged_kwargs, which includes user kwargs.

        This validates the PR change: user kwargs are merged into module_kwargs
        *before* set_up_systems is called (not after, as was the case before).
        """
        env = self.env
        captured = {}

        class CapturingMachine:
            environment = env

            @staticmethod
            def get_system(name):
                return name

            @staticmethod
            def set_up_systems(systems, proc):
                # Snapshot keys and values at entry, before any modification.
                captured.update(systems)
                for q in list(systems):
                    systems[q] = proc.q(q)

            @staticmethod
            def clean_up_systems(systems):
                pass

        @qcdl(machine=CapturingMachine)
        def main(q0, q1, **kwargs):
            q0.x()
            q0.measure()

        main(my_extra=42)
        assert "my_extra" in captured
        assert captured["my_extra"] == 42

    def test_machine_clean_up_receives_merged_kwargs(self):
        """clean_up_systems is called with merged_kwargs, including user kwargs."""
        env = self.env
        cleanup_received = {}

        class CapturingMachine:
            environment = env

            @staticmethod
            def get_system(name):
                return name

            @staticmethod
            def set_up_systems(systems, proc):
                for q in list(systems):
                    systems[q] = proc.q(q)

            @staticmethod
            def clean_up_systems(systems):
                cleanup_received.update(systems)

        @qcdl(machine=CapturingMachine)
        def main(q0, q1, **kwargs):
            q0.x()
            q0.measure()

        main(my_extra=99)
        assert "my_extra" in cleanup_received

    def test_machine_uses_environment_from_machine(self):
        """The environment used for the circuit is taken from machine.environment."""
        env = FakeEnv(["q0", "q1", "q2"])
        machine, tracker = make_fake_machine(env)

        @qcdl(machine=machine)
        def main(q0, q1, q2):
            q0.x()
            q0.measure()

        main()
        # get_system should have been called for all three modules
        assert set(tracker.get_system_calls) == {"q0", "q1", "q2"}


# ---------------------------------------------------------------------------
# 3. f_keywords-based kwargs filtering
# ---------------------------------------------------------------------------


class TestQCDLKwargsFiltering:
    """Tests for how @qcdl decides which kwargs to pass to the circuit function.

    If f has no **kwargs, only keys matching f's explicit argument names are
    forwarded (passed_kwargs). If f has **kwargs, all merged_kwargs are forwarded.
    """

    def test_no_varkw_filters_out_extra_modules(self):
        """Without **kwargs, qubits not named in f's signature are excluded."""
        env = FakeEnv(["q0", "q1", "q2"])
        received = {}

        @qcdl(environment=env)
        def main(q0):
            received["q0"] = q0
            q0.x()
            q0.measure()

        main()
        # Only q0 should have been passed; q1 and q2 must be absent.
        assert set(received) == {"q0"}

    def test_varkw_receives_all_modules(self):
        """With **kwargs, all modules from the environment are forwarded."""
        env = FakeEnv(["q0", "q1", "q2"])
        received = {}

        @qcdl(environment=env)
        def main(**kwargs):
            received.update(kwargs)
            kwargs["q0"].x()
            kwargs["q0"].measure()

        main()
        assert set(received) == {"q0", "q1", "q2"}

    def test_no_varkw_receives_user_kwarg_present_in_sig(self):
        """A caller kwarg whose name appears in f's signature is passed through."""
        received = {}

        @qcdl()
        def main(q0, angle=None):
            received["angle"] = angle
            q0.x()
            q0.measure()

        main(angle=2.0)
        assert received["angle"] == 2.0

    def test_no_varkw_excludes_user_kwarg_absent_from_sig(self):
        """A caller kwarg whose name is NOT in f's signature is filtered out."""

        @qcdl(1)
        def main(q0):
            q0.x()
            q0.measure()

        # Passing an unexpected kwarg should not raise; it is simply dropped.
        result = main(unknown_param=99)
        assert isinstance(result, QCDLProgram)

    def test_varkw_receives_user_kwargs_alongside_qubits(self):
        """With **kwargs, caller kwargs are forwarded alongside qubits."""
        received = {}

        @qcdl(2)
        def main(**kwargs):
            received.update(kwargs)
            kwargs["q0"].x()
            kwargs["q0"].measure()

        main(my_param=42)
        assert "q0" in received
        assert "q1" in received
        assert received["my_param"] == 42

    def test_machine_no_varkw_filters_extra_modules(self):
        """In machine mode, extra modules not in f's signature are excluded."""
        env = FakeEnv(["q0", "q1", "q2"])
        machine, _ = make_fake_machine(env)
        received = {}

        @qcdl(machine=machine)
        def main(q0):
            received["q0"] = q0
            q0.x()
            q0.measure()

        main()
        assert set(received) == {"q0"}

    def test_machine_varkw_receives_all_modules(self):
        """In machine mode with **kwargs, all modules are forwarded to f."""
        env = FakeEnv(["q0", "q1", "q2"])
        machine, _ = make_fake_machine(env)
        received = {}

        @qcdl(machine=machine)
        def main(**kwargs):
            received.update(kwargs)
            kwargs["q0"].x()
            kwargs["q0"].measure()

        main()
        assert set(received) == {"q0", "q1", "q2"}


# ---------------------------------------------------------------------------
# 4. User kwargs merging
# ---------------------------------------------------------------------------


class TestQCDLUserKwargsMerging:
    """User-supplied kwargs are merged into module_kwargs before being passed."""

    def test_user_kwargs_reach_circuit_body(self):
        """Keyword arguments from the caller are forwarded to f."""
        received = {}

        @qcdl(1)
        def main(q0, angle=None):
            received["angle"] = angle
            q0.x()
            q0.measure()

        main(angle=1.23)
        assert received["angle"] == 1.23

    def test_user_kwargs_override_auto_created_qubit(self):
        """A caller kwarg with the same name as an auto-qubit overrides it."""
        sentinel = object()
        received = {}

        @qcdl(1)
        def main(q0, **kwargs):
            received["q0"] = q0

        main(q0=sentinel)
        assert received["q0"] is sentinel

    def test_positional_args_forwarded_to_f(self):
        """Positional args passed to the wrapper are forwarded to f."""
        received = {}

        @qcdl(1)
        def main(angle, q0):
            received["angle"] = angle
            q0.x()
            q0.measure()

        main(0.75)
        assert received["angle"] == 0.75

    def test_callable_object_as_decorated_function(self):
        """@qcdl works with any callable, not just plain functions."""
        received = {}

        class MyCircuit:
            def __call__(self, q0, q1, my_kwarg=None):
                received["my_kwarg"] = my_kwarg
                q0.x()
                q0.measure()

        decorated = qcdl()(MyCircuit())
        result = decorated(my_kwarg="hello")
        assert isinstance(result, QCDLProgram)
        assert received["my_kwarg"] == "hello"


class TestQCDLCircuitStateAndOutputs:
    """Tests for QCDLCircuit stateful internals and output resource handling."""

    def test_output_resources(self):
        qcdl_obj = QCDLCircuit()

        class Mock:
            pass

        q = Mock()
        q.qcdl_module_name = "q0"

        q1 = Mock()
        q1.qcdl_module_name = "q0"

        for category in ["GOF", "DYN"]:
            for _ in range(3):
                with pytest.raises(QCDLUserError):
                    qcdl_obj.release_output(q, "DYN1")

                with pytest.raises(QCDLUserError):
                    qcdl_obj.reserve_output(q, name="something random")
                with pytest.raises(QCDLUserError):
                    qcdl_obj.reserve_output(q, category="something random")

                reserved = []
                num = 3 if category == "DYN" else 4
                for _ in range(num):
                    out = qcdl_obj.reserve_output(q, category=category)
                    with pytest.raises(QCDLUserError):
                        qcdl_obj.reserve_output(q, name=out)
                    assert out is not None
                    reserved.append(out)

                    intersection = qcdl_obj.available_outputs([q, q1], category)
                    assert out not in intersection

                with pytest.raises(QCDLUserError):
                    qcdl_obj.reserve_output(q, category)

                for out in reserved:
                    qcdl_obj.release_output(q, out)

                with pytest.raises(QCDLUserError):
                    qcdl_obj.release_output(q, "something random")

    def test_restore_next_index(self):
        idx_name = "my index"
        num_first = 4
        num_second = 3

        @qcdl(1)
        def first(q0):
            for idx in range(num_first):
                assert q0.state.get_next_index(idx_name) == idx

        qcdl_json1 = first()
        _check_serializable(qcdl_json1)
        assert qcdl_json1.next_indices[idx_name] == num_first

        @qcdl(1, next_indices=qcdl_json1.next_indices)
        def second(q0):
            for idx in range(num_second):
                assert q0.state.get_next_index(idx_name) == idx + num_first

        qcdl_json2 = second()
        assert qcdl_json2.next_indices[idx_name] == num_first + num_second


class TestQCDLDecoratorModes:
    """Tests decorator mode equivalence across function and method callables."""

    def test_qcdl_decorator_equivalent_function_modes(self):
        rand_num = 1234

        @qcdl(3)
        def main_env(q0, q1, q2, my_kwarg=None, **kwargs):
            assert my_kwarg in [1, rand_num]
            _qcdl_decorator_reference(q0, q1, q2, my_kwarg)

        @qcdl(num_qubits=4)
        def main_num(q0, q1, q2, my_kwarg=None, **kwargs):
            assert my_kwarg in [2, rand_num]
            _qcdl_decorator_reference(q0, q1, q2, my_kwarg)

        @qcdl()
        def main_infer(q0, q1, q2, my_kwarg=None, **kwargs):
            assert my_kwarg in [3, rand_num]
            _qcdl_decorator_reference(q0, q1, q2, my_kwarg)

        qcdl_json = main_env(my_kwarg=1)
        _check_serializable(qcdl_json)
        _assert_statement_count(qcdl_json)

        _assert_statement_count(main_num(my_kwarg=2))
        _assert_statement_count(main_infer(my_kwarg=3))

        assert main_num(my_kwarg=rand_num) == main_infer(my_kwarg=rand_num)
        assert main_env(my_kwarg=rand_num) == main_infer(my_kwarg=rand_num)

    def test_qcdl_decorator_class_method_mode(self):
        rand_num = 1234

        class QCDLDec(object):
            @qcdl()
            def sequence(self, q0, q1, q2, my_kwarg=None, **kwargs):
                assert isinstance(self, QCDLDec)
                assert my_kwarg in [4, rand_num]
                _qcdl_decorator_reference(q0, q1, q2, my_kwarg)

        @qcdl(3)
        def main_env(q0, q1, q2, my_kwarg=None, **kwargs):
            _qcdl_decorator_reference(q0, q1, q2, my_kwarg)

        obj = QCDLDec()
        qcdl_json = obj.sequence(my_kwarg=4)
        _check_serializable(qcdl_json)
        _assert_statement_count(qcdl_json)
        assert main_env(my_kwarg=rand_num) == obj.sequence(my_kwarg=rand_num)


class TestQCDLNumQubitsValidation:
    """num_qubits has to be able to produce qubits."""

    @pytest.mark.parametrize("num_qubits", [0, -1, -10])
    def test_num_qubits_below_one_raises(self, num_qubits):
        with pytest.raises(QCDLUserError, match="must be at least 1"):
            qcdl(num_qubits)

    @pytest.mark.parametrize("num_qubits", [2.0, 2.5, "2", True, [2]])
    def test_num_qubits_must_be_an_integer(self, num_qubits):
        with pytest.raises(QCDLUserError, match="must be an integer"):
            qcdl(num_qubits)

    def test_decorator_used_without_calling_it_raises(self):
        """A bare @qcdl passes the function in as num_qubits."""

        def main(q0):
            pass

        with pytest.raises(QCDLUserError, match="must be called"):
            qcdl(main)

    def test_num_qubits_of_one_is_allowed(self):
        @qcdl(1)
        def main(q0):
            q0.measure()

        assert [str(q) for q in main().program.signature.qubits_used] == ["q0"]


class TestQCDLSignatureValidation:
    """The signature of the decorated function has to match the qubits made.

    Qubits are injected by keyword, so a signature that does not name them all
    either loses a qubit or leaves a parameter unbound. Both were silent.
    """

    def test_generated_qubit_with_no_parameter_raises(self):
        @qcdl(3)
        def main(q0, q1):
            q0.h()

        with pytest.raises(QCDLUserError, match="no parameter for q2"):
            main()

    def test_every_generated_qubit_named_is_accepted(self):
        @qcdl(3)
        def main(q0, q1, q2, my_angle=0):
            q0.h()

        assert [str(q) for q in main(my_angle=0.5).program.signature.qubits_used] == [
            "q0"
        ]

    def test_var_keyword_signature_absorbs_every_qubit(self):
        @qcdl(3)
        def main(**kwargs):
            assert set(kwargs) == {"q0", "q1", "q2"}
            kwargs["q0"].h()

        main()

    def test_environment_may_supply_more_qubits_than_the_signature(self):
        """The environment supplies its whole set, so dropping is expected."""
        env = FakeEnv(["q0", "q1", "q2"])

        @qcdl(environment=env)
        def main(q0):
            q0.measure()

        assert isinstance(main(), QCDLProgram)

    def test_parameter_not_named_qN_explains_the_rule(self):
        @qcdl(1)
        def main(alpha):
            pass

        with pytest.raises(TypeError) as excinfo:
            main()

        message = str(excinfo.value)
        # keep python's own wording, then say why nothing was passed
        assert "missing 1 required positional argument: 'alpha'" in message
        assert "q0, q1, ..." in message
        assert "'alpha' does not match" in message

    def test_qubit_parameter_beyond_num_qubits_explains_the_rule(self):
        @qcdl(2)
        def main(q0, q1, q2):
            pass

        with pytest.raises(TypeError) as excinfo:
            main()

        message = str(excinfo.value)
        assert "missing 1 required positional argument: 'q2'" in message
        assert "supplied q0, q1" in message
        assert "num_qubits" in message

    def test_several_unfilled_parameters_are_reported_together(self):
        @qcdl(1)
        def main(q0, alpha, beta):
            pass

        with pytest.raises(
            TypeError, match="missing 2 required positional arguments: 'alpha', 'beta'"
        ):
            main()

    def test_unfilled_parameter_may_be_passed_by_the_caller(self):
        @qcdl(1)
        def main(q0, alpha):
            q0.comment(str(alpha))
            q0.measure()

        assert isinstance(main(alpha=3), QCDLProgram)

    def test_keyword_only_parameter_without_a_default_is_reported(self):
        @qcdl(1)
        def main(q0, *, alpha):
            pass

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            main()

    def test_inferred_mode_never_drops_or_starves_a_parameter(self):
        @qcdl()
        def main(q0, q5, alpha=1):
            q0.measure()
            q5.measure()

        assert [str(q) for q in main().program.signature.qubits_used] == ["q0", "q5"]


def test_fspec():
    def my_meth1(q0, q1, q2=321):
        pass

    args, keys = _get_fspec(my_meth1)
    assert args == ["q0", "q1", "q2"]
    assert keys is None

    def my_meth2(q0, q1, q2=321, **my_kwargs):
        pass

    args, keys = _get_fspec(my_meth2)
    assert args == ["q0", "q1", "q2"]
    assert keys == "my_kwargs"

    class Expt(object):
        def my_expt_meth(self, q0, q1, q2=123):
            pass

        def __call__(self, q0, q1, q3=111):
            pass

    obj = Expt()
    args, keys = _get_fspec(obj.my_expt_meth)
    assert args == ["self", "q0", "q1", "q2"]
    assert keys is None

    args, keys = _get_fspec(obj)
    assert args == ["self", "q0", "q1", "q3"]
    assert keys is None
