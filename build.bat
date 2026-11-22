@echo off
setlocal
rem Build main.cu -> lib\dft.dll with nvcc (loaded from dft.py via ctypes).
rem nvcc needs cl.exe and the MSVC environment, so load vcvars64 first if cl isn't on PATH.

cd /d "%~dp0"

where cl >nul 2>nul
if errorlevel 1 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
    if errorlevel 1 (
        echo Failed to load the MSVC environment.
        exit /b 1
    )
)

if not exist lib mkdir lib

rem sm_120 = RTX 50-series (Blackwell). Pass "debug" for a debug build: build.bat debug
if /i "%~1"=="debug" (
    set "FLAGS=-G -g -O0"
) else (
    set "FLAGS=-O3"
)

nvcc %FLAGS% -std=c++17 -arch=sm_120 -shared main.cu -o lib\dft.dll
exit /b %errorlevel%
exit /b %errorlevel%
