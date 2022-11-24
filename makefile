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
