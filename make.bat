@echo off

set PYTHON=python3
set report=html
set COVERAGE=--cov=dwave.gate --cov-report=%report%
set TESTRUNNER=-m pytest tests

IF /I "%1"=="install" GOTO install
IF /I "%1"=="clean" GOTO clean
IF /I "%1"=="test" GOTO test
IF /I "%1"=="coverage" GOTO coverage
IF /I "%1"=="docs" GOTO docs
IF /I "%1"=="clean-docs" GOTO clean-docs
IF /I "%1"=="format" GOTO format
GOTO error

:install
	CALL %PYTHON% -m pip install .
	CALL %PYTHON% .\dwave\gate\simulator\operation_generation.py
	CALL %PYTHON% setup.py build_ext --inplace
	GOTO :EOF

:clean
	IF EXIST ".pytest_cache/" RMDIR /S /Q ".pytest_cache/"
	IF EXIST "dwave_gate.egg-info/" RMDIR /S /Q "dwave_gate.egg-info/"
	IF EXIST "dist/" RMDIR /S /Q "dist/"
	IF EXIST "build/" RMDIR /S /Q "build/"
GOTO :EOF

:test
	CALL %PYTHON% %TESTRUNNER%
	GOTO :EOF

:coverage
	CALL %PYTHON% %TESTRUNNER% %COVERAGE%
	GOTO :EOF

:docs
	CALL .\docs\make.bat html
	GOTO :EOF

:clean-docs
	CALL .\docs\make.bat clean
	GOTO :EOF

:format
	CALL black -l 100 .\dwave\gate .\tests
	CALL isort -l 100 --profile black .\dwave\gate .\tests
	GOTO :EOF

:error
    IF "%1"=="" (
        ECHO make: *** No targets specified and no makefile found.  Stop.
    ) ELSE (
        ECHO make: *** No rule to make target '%1%'. Stop.
    )
    GOTO :EOF
