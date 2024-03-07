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

"""Unicode characters used to draw circuits in the terminal."""

__all__ = [
    "lang",
    "rang",
    "vbar",
    "hbar",
    "crs",
    "tle",
    "tri",
    "ble",
    "bri",
    "lba",
    "rba",
    "tco",
    "bco",
    "cpl",
    "blk",
    "tbk",
    "bbk",
    "swp",
    "spc",
]

lang = "\u27E8"  # ⟨
# rang = u'\u2771'  # ⟩
# rang = "\u276F"  # ⟩
rang = "\u27E9"  #     ⟩
# vbar = u'\u007C'  # │
vbar = "\u2502"  # │
hbar = "\u2500"  # ─

crs = "\u253C"  # ┼
tle = "\u250C"  # ┌
tri = "\u2510"  # ┐
ble = "\u2514"  # └
bri = "\u2518"  # ┘

lba = "\u2524"  # ┤
rba = "\u251C"  # ├
tco = "\u2534"  # ┴
bco = "\u252C"  # ┬

cpl = "\u2295"  # ⊕
blk = "\u2588"  # █
tbk = "\u2584"  # ▄
bbk = "\u2580"  # ▀

swp = "\u2573"  # ╳
# spc = u'\u2007'  # space
spc = "\u0020"  # space
