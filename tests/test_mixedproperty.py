# Copyright 2022 D-Wave Systems Inc.
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

from dwave.gate.mixedproperty import mixedproperty


class MixedpropertyExample:
    """Dummy class to test the ``mixedproperty`` decorator."""

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


@pytest.fixture(scope="function")
def mixedpropertyexample():
    """Returns a instance of the MixedpropertyExample class."""
    return MixedpropertyExample()


class TestMixedPropery:
    """Unittests for the mixedproperty class/decorator."""

    def test_with_class(self, mixedpropertyexample):
        """Test a mixedproperty with access only to the class."""
        assert mixedpropertyexample.a == "a_class"
        assert MixedpropertyExample.a == "a_class"

    def test_with_instance(self, mixedpropertyexample):
        """Test a mixedproperty with access to the class and instance."""
        assert mixedpropertyexample.b == "b_instance"
        assert MixedpropertyExample.b == "b_class"

    def test_with_self_required(self, mixedpropertyexample):
        """Test a mixedproperty with access only to the instance."""
        assert mixedpropertyexample.c == "c_instance"
        assert MixedpropertyExample.c is None

    def test_set_mixedproperty_with_class(self, mixedpropertyexample):
        """Test that the correct error is raised when setting a mixedproperty with access only to the class."""
        with pytest.raises(AttributeError, match="can't set attribute"):
            mixedpropertyexample.a = "apple"

    def test_set_mixedproperty_with_instance(self, mixedpropertyexample):
        """Test that the correct error is raised when setting a mixedproperty with access to the class and instance."""
        with pytest.raises(AttributeError, match="can't set attribute"):
            mixedpropertyexample.b = "banana"

    def test_set_mixedproperty_with_self_required(self, mixedpropertyexample):
        """Test that the correct error is raised when setting a mixedproperty with access only to the instance."""
        with pytest.raises(AttributeError, match="can't set attribute"):
            mixedpropertyexample.c = "coconut"
