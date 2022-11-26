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

PYTHON3 := $(shell which python3 2>/dev/null)
BLACK := $(shell which black 2>/dev/null)
ISORT := $(shell which isort 2>/dev/null)

PYTHON := python3

report ?= html  # HTML is the default report type
COVERAGE := --cov=dwave.gate --cov-report=$(report)

TESTRUNNER := -m pytest tests

.PHONY: install
install:
ifndef PYTHON3
	@echo "D-Wave Gate Model software requires at least Python 3.7"
endif
	$(PYTHON) setup.py build_ext --inplace

# whether coverage files should be removed (true)
# default is to not delete coverage files (false)
cov := false

.PHONY: clean
clean:
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf dist/ build/

ifeq ($(cov),true)
	rm -rf coverage_html/
	rm -f .coverage coverage.*
endif

.PHONY: test
test:
	$(PYTHON) $(TESTRUNNER)

.PHONY: coverage
coverage:
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)

.PHONY: format
format:
ifndef ISORT
	@echo "D-Wave Gate Model software uses isort to sort imports."
endif
	isort -l 100 --profile black ./dwave/gate ./tests
ifndef BLACK
	@echo "D-Wave Gate Model software uses the Black formatter."
endif
	black -l 100 ./dwave/gate ./tests
