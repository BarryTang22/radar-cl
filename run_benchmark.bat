@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM Full PGSU Benchmark
REM 7 streams x compatible methods x 3 seeds = 177 runs
REM Reports: A_f, A_bar, F (mean +/- std over 3 seeds)
REM ============================================================

set SEEDS=42 123 456
set EPOCHS=50
set SCRIPT=trainings\train_cl.py
set COUNT=0
set TOTAL=177

echo ============================================================
echo  PGSU Benchmark: %TOTAL% runs (%EPOCHS% epochs, 3 seeds)
echo  Started: %date% %time%
echo ============================================================

REM ----------------------------------------------------------
REM Stream 1: DMM-S (CNN, ResNet18) - 7 methods
REM   Skip: ease (not class-incr), l2p/coda/dualprompt (transformer-only)
REM ----------------------------------------------------------
for %%A in (naive ewc lwf replay derpp co2l pgsu) do (
    set /a COUNT+=3
    echo.
    echo [!COUNT!/%TOTAL%] dmm / scene / %%A
    python %SCRIPT% --dataset dmm --setting scene --algorithm %%A --epochs %EPOCHS% --seeds %SEEDS%
    if errorlevel 1 echo ERROR: dmm/scene/%%A failed >> benchmark_errors.log
)

REM ----------------------------------------------------------
REM Stream 2: DRC-S (Transformer) - 10 methods
REM   Skip: ease (not class-incr)
REM ----------------------------------------------------------
for %%A in (naive ewc lwf replay derpp co2l l2p coda dualprompt pgsu) do (
    set /a COUNT+=3
    echo.
    echo [!COUNT!/%TOTAL%] drc / scene / %%A
    python %SCRIPT% --dataset drc --setting scene --algorithm %%A --epochs %EPOCHS% --seeds %SEEDS%
    if errorlevel 1 echo ERROR: drc/scene/%%A failed >> benchmark_errors.log
)

REM ----------------------------------------------------------
REM Stream 3: CI4R-F (CNN, ResNet18) - 7 methods
REM   Skip: ease (not class-incr), l2p/coda/dualprompt (transformer-only)
REM ----------------------------------------------------------
for %%A in (naive ewc lwf replay derpp co2l pgsu) do (
    set /a COUNT+=3
    echo.
    echo [!COUNT!/%TOTAL%] ci4r / frequency / %%A
    python %SCRIPT% --dataset ci4r --setting frequency --algorithm %%A --epochs %EPOCHS% --seeds %SEEDS%
    if errorlevel 1 echo ERROR: ci4r/frequency/%%A failed >> benchmark_errors.log
)

REM ----------------------------------------------------------
REM Stream 4: CI4R-C (CNN, ResNet18) - 8 methods
REM   Skip: l2p/coda/dualprompt (transformer-only)
REM ----------------------------------------------------------
for %%A in (naive ewc lwf replay derpp co2l ease pgsu) do (
    set /a COUNT+=3
    echo.
    echo [!COUNT!/%TOTAL%] ci4r / class / %%A
    python %SCRIPT% --dataset ci4r --setting class --algorithm %%A --epochs %EPOCHS% --seeds %SEEDS%
    if errorlevel 1 echo ERROR: ci4r/class/%%A failed >> benchmark_errors.log
)

REM ----------------------------------------------------------
REM Stream 5: CI4R-M (CNN, ResNet18) - 8 methods
REM   Skip: l2p/coda/dualprompt (transformer-only)
REM ----------------------------------------------------------
for %%A in (naive ewc lwf replay derpp co2l ease pgsu) do (
    set /a COUNT+=3
    echo.
    echo [!COUNT!/%TOTAL%] ci4r / mixed / %%A
    python %SCRIPT% --dataset ci4r --setting mixed --algorithm %%A --epochs %EPOCHS% --seeds %SEEDS%
    if errorlevel 1 echo ERROR: ci4r/mixed/%%A failed >> benchmark_errors.log
)

REM ----------------------------------------------------------
REM Stream 6: RadHAR-C (Transformer, Conv3D) - 11 methods
REM ----------------------------------------------------------
for %%A in (naive ewc lwf replay derpp co2l ease l2p coda dualprompt pgsu) do (
    set /a COUNT+=3
    echo.
    echo [!COUNT!/%TOTAL%] radhar / class / %%A
    python %SCRIPT% --dataset radhar --setting class --algorithm %%A --epochs %EPOCHS% --seeds %SEEDS%
    if errorlevel 1 echo ERROR: radhar/class/%%A failed >> benchmark_errors.log
)

REM ----------------------------------------------------------
REM Stream 7: DIAT-C (CNN, ResNet18) - 8 methods
REM   Skip: l2p/coda/dualprompt (transformer-only)
REM ----------------------------------------------------------
for %%A in (naive ewc lwf replay derpp co2l ease pgsu) do (
    set /a COUNT+=3
    echo.
    echo [!COUNT!/%TOTAL%] diat / class / %%A
    python %SCRIPT% --dataset diat --setting class --algorithm %%A --epochs %EPOCHS% --seeds %SEEDS%
    if errorlevel 1 echo ERROR: diat/class/%%A failed >> benchmark_errors.log
)

echo.
echo ============================================================
echo  Benchmark complete: %date% %time%
echo  Results: results\cl\cl_results.csv
if exist benchmark_errors.log (
    echo  ERRORS logged in benchmark_errors.log
) else (
    echo  All 177 runs finished without errors
)
echo ============================================================
