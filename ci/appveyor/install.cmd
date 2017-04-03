@echo on

:: Activate test environment anaconda
set PATH=%MINICONDA%;%MINICONDA%\\Scripts;%PATH%
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -q -n test-environment python=%PYTHON_VERSION% numpy scipy nose
activate test-environment

:: Compile osqp
mkdir build
cd build
cmake -G"%GENERATOR%" ..
cmake --build .


@echo off