# Confidential & Proprietary Information: D-Wave Systems Inc.
from __future__ import annotations

import inspect
import uuid
from abc import abstractmethod
from typing import Any, Callable, Optional


class classproperty:
    """Decorator class for creating a property for a class method.

    Note that, apart from being much simplified, the ``classproperty`` decorator
    differs from the regular ``property`` decorator by supporting access to both
    the class and the instance (the latter is only available when calling the
    method on an instance of the class). The signature parameters are positional
    and assume that the first parameter is the class and the (optional) second
    parameter is the instance.

    This property can optionally accept parameters (either positional or keyword
    arguments) when initialized. Allowed parameters are listed under 'Args'
    below.

    Args:
        self_required: Whether ``self`` is required for the property to be able
            to return the requested property, i.e., the function will only work
            when called on an instance and not on the class.

    Example:

        .. code-block:: python

            class ClasspropertyExample:
                _a = "a_class"
                _b = "b_class"

                def __init__(self):
                    self._b = "b_instance"
                    self._c = "c_instance"

                # can be called on both instance and class and will
                # return the same result regardless
                @classproperty
                def a(cls):
                    return cls._a

                # can be called on both instance and class and will
                # return different results (e.g., different defaults)
                @classproperty
                def b(cls, self):
                    if self:
                        return self._b
                    return cls._b

                # can be called on both instance and class and will
                # return 'None' if called on class
                @classproperty(self_required=True)
                def c(cls, self):
                    return self._c
    """

    def __init__(self, *args, **kwargs):
        self._self_required: bool = kwargs.get("self_required", False)
        if args and isinstance(args[0], Callable):
            self.__call__(args[0])

    def __call__(self, callable: Callable) -> classproperty:
        """Initialization of the decorated function.

        Args:
            callable: Method decorated by the ``classproperty`` decorator."""
        self._callable = callable
        return self

    def __get__(self, instance: Optional[object], cls: type) -> Any:
        """Return the internal function call.

        Depending on whether the internal function signature contains 2
        parameters (class and instance object; normally `cls` and `self`) or
        only 1 parameter (only class; normally `cls`), different function calls
        are returned.

        Args:
            instance: Instance object if called on such.
            cls: Class or class of instance.

        Returns:
            Any: Output of the decorated method, if any.
        """
        # if abstract method is called during class construction, return 'None'
        if getattr(self._callable, "__isabstractmethod__", False):
            return self._callable

        num_parameters = len(inspect.signature(self._callable).parameters)

        # if called on class while requiring access to 'self', return 'None'
        if self._self_required and num_parameters >= 2 and instance is None:
            return None

        if num_parameters == 1:
            return self._callable(cls)

        return self._callable(cls, instance)


class abstractclassproperty(classproperty):
    """Decorator class to support abstract ``classproperty`` methods.

    Args:
        callable: Method decorated by the ``abstractclassproperty`` decorator.
    """

    __isabstractmethod__: bool = True
    """Used internally by Python to keep track of abstract methods."""

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


def generate_id(prefix: str = None, k: int = 6) -> str:
    """Generate a random ID label.

    A random string of integers with an optional prefix.

    Args:
        prefix: Prefix for the ID label. Can be any string except
            ``"qubit"``/``"bit"``, in which case the prefix defaults to
            ``'q'``/``'c'``.
        k: Number of integers in the ID, excluding the prefix.

    Returns:
        str: Random ID string label.
    """
    reg_id = str(uuid.uuid4().int)[:k]

    if prefix == "qubit":
        return "q" + reg_id
    elif prefix == "bit":
        return "c" + reg_id

    if prefix is None:
        return reg_id
    return prefix + reg_id


class IntegerCounter:
    """Integer counter to generate and keep track of labels.

    Args:
        start: Integer index at which to start the counting.
        prefix: Prefix for the ID label (defaults to none).
    """

    def __init__(self, start: int = 0, prefix: str = None) -> None:
        self._start = start
        self._counter = start - 1
        self._prefix = prefix or ""

    def reset(self, start: int = None) -> None:
        """Resets the counter to restart at the initialized starting index.

        Args:
            start: Integer index at which to start the counting.
        """
        if start:
            self._start = start
        self._counter = self._start - 1

    @property
    def counter(self) -> int:
        """Current counter ID label."""
        return self._counter

    def increment(self, n: int = 1) -> None:
        """Increment the counter without returning a label.

        Args:
            n: The number of steps to increment the counter with.
        """
        self._counter += n

    def next(self) -> str:
        """Increment the counter with 1, returning the ID label.

        Returns:
            str: The next ID label.
        """
        self._counter += 1
        return self._prefix + str(self._counter)


#####################
# Custom exceptions #
#####################


class CircuitError(Exception):
    """Exception to be raised when there is an error with a Circuit."""


class TemplateError(Exception):
    """Exception to be raised when there is an error with a Template."""
