# Confidential & Proprietary Information: D-Wave Systems Inc.
from __future__ import annotations

import inspect
from typing import Any, Callable, Optional


class mixedproperty:
    """Decorator class for creating a property for a class/instance method.

    Note that the ``mixedproperty`` decorator differs from the regular
    ``property`` decorator by supporting access to both the class (``cls``) and
    the instance (``self``). The latter is only available when calling the
    method on an instance of the class. The signature parameters are positional
    and assume that the first parameter is the class and the second parameter
    (optional) is the instance.

    This property can optionally also accept parameters (either positional or
    keyword) when initialized. Allowed parameters are listed under 'Args' below.

    Args:
        self_required: Whether ``self`` is required for the property to be able
            to return the requested property, i.e., the function will only work
            when called on an instance and not on the class.

    Example:

        .. code-block:: python

            class MixedpropertyExample:
                _a = "a_class"
                _b = "b_class"

                def __init__(self):
                    self._b = "b_instance"
                    self._c = "c_instance"

                # can be called on both instance and class and will
                # return the same result regardless
                @mixedproperty
                def a(cls):
                    return cls._a

                # can be called on both instance and class and will
                # return different results (e.g., different defaults)
                @mixedproperty
                def b(cls, self):
                    if self:
                        return self._b
                    return cls._b

                # can be called on both instance and class and will
                # return 'None' if called on class
                @mixedproperty(self_required=True)
                def c(cls, self):
                    return self._c
    """

    def __init__(self, *args, **kwargs):
        self._self_required: bool = kwargs.pop("self_required", False)
        if args and isinstance(args[0], Callable):
            self.__call__(args[0])

    def __call__(self, callable: Callable) -> mixedproperty:
        """Initialization of the decorated function.

        Args:
            callable: Method decorated by the ``mixedproperty`` decorator."""
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

    def __set__(self, instance: Optional[object], value: Any) -> None:
        raise AttributeError("can't set attribute")


class abstractmixedproperty(mixedproperty):
    """Decorator class to support abstract ``mixedproperty`` methods.

    Args:
        callable: Method decorated by the ``abstractmixedproperty`` decorator.
    """

    __isabstractmethod__: bool = True
    """Used internally by Python to keep track of abstract methods."""

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)
