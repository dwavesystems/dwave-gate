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

from dwave.gate.tools.counters import IDCounter


class TestIDCounter:
    """Unit tests for the ``IDCounter`` class"""

    def test_get_ID(self, monkeypatch):
        """Test retrieving a random ID number."""
        monkeypatch.setattr(IDCounter, "_default_length", 4)
        monkeypatch.setattr(IDCounter, "_default_batch", 1000)
        IDCounter.reset()

        assert len(IDCounter.id_set) == 0

        id_0 = IDCounter.next()
        assert len(id_0) == 4
        assert len(IDCounter.id_set) == 1000 - 1

        id_1 = IDCounter.next()
        assert id_0 != id_1
        assert len(IDCounter.id_set) == 1000 - 2

    def test_auto_refresh_set(self, monkeypatch, mocker):
        """Test automatically refreshing set (without incrementing length)."""
        monkeypatch.setattr(IDCounter, "_default_length", 4)
        # set small batch size to generate new batch quickly
        monkeypatch.setattr(IDCounter, "_default_batch", 2)
        IDCounter.reset()

        spy = mocker.spy(IDCounter, "refresh")

        assert spy.call_count == 0
        IDCounter.next()
        assert spy.call_count == 1
        IDCounter.next()
        assert spy.call_count == 1
        IDCounter.next()
        assert spy.call_count == 2
        IDCounter.next()
        assert spy.call_count == 2
        IDCounter.next()
        assert spy.call_count == 3

    def test_auto_refresh_length(self, monkeypatch, mocker):
        """Test automatically refreshing set and incrementing length."""
        # set small length to make sure that length is incremented
        monkeypatch.setattr(IDCounter, "_default_length", 1)
        # set small batch size to make sure that length=2 is sufficient
        monkeypatch.setattr(IDCounter, "_default_batch", 50)
        IDCounter.reset()

        spy = mocker.spy(IDCounter, "refresh")

        assert spy.call_count == 0
        assert IDCounter.length == 1

        for _ in range(len(IDCounter._alphanum)):
            IDCounter.next()

        assert spy.call_count == 1
        assert IDCounter.length == 2

    def test_manual_refresh_set(self, monkeypatch):
        """Test automatically refreshing set (without incrementing length)."""
        monkeypatch.setattr(IDCounter, "_default_length", 4)
        monkeypatch.setattr(IDCounter, "_default_batch", 1000)
        IDCounter.reset()

        assert len(IDCounter.id_set) == 0
        IDCounter.next()
        assert len(IDCounter.id_set) == 1000 - 1

        IDCounter.refresh()
        assert len(IDCounter.id_set) == 1000 - 1 + 1000

    def test_manual_refresh_length(self, monkeypatch):
        """Test manually refreshing set and and incrementing length."""
        monkeypatch.setattr(IDCounter, "_default_length", 1)
        monkeypatch.setattr(IDCounter, "_default_batch", 20)
        IDCounter.reset()

        assert IDCounter.length == 1
        # implicit (first) refresh requesting 20 new IDs
        IDCounter.next()
        # explicit refresh requestin 20 new IDs (> `len(IDCounter._alphanums`))
        # causing the length to increment to 2
        IDCounter.refresh()
        assert IDCounter.length == 2

    def test_reset(self, monkeypatch):
        """Test resetting the ID counter."""
        monkeypatch.setattr(IDCounter, "_default_length", 2)
        monkeypatch.setattr(IDCounter, "_default_batch", 1000)
        IDCounter.reset()

        assert IDCounter.length == 2
        assert len(IDCounter.id_set) == 0

        for _ in range(60):
            IDCounter.next()

        assert IDCounter.length != 2
        assert len(IDCounter.id_set) != 0

        IDCounter.reset()

        assert IDCounter.length == 2
        assert len(IDCounter.id_set) == 0

    def test_insufficient_chars_warning(self, monkeypatch):
        """Test that a warning is raised when having an insufficient number of characters."""
        monkeypatch.setattr(IDCounter, "_default_length", 4)
        monkeypatch.setattr(IDCounter, "_default_batch", 1000)
        monkeypatch.setattr(IDCounter, "_alphanum", "abcde")
        IDCounter.reset()

        with pytest.warns(match="Insufficient characters"):
            IDCounter.next()

    def test_insufficient_chars_error(self, monkeypatch):
        """Test that an error is raised when having less characters than requested length."""
        monkeypatch.setattr(IDCounter, "_default_length", 4)
        monkeypatch.setattr(IDCounter, "_default_batch", 1000)
        # have less available chars than length
        monkeypatch.setattr(IDCounter, "_alphanum", "ab")
        IDCounter.reset()

        with pytest.raises(
            ValueError,
            match="ID length cannot be longer than number of unique characters available.",
        ):
            IDCounter.next()
