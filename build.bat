@echo off
REM VPU MoE Lite - Build and Run Script
REM Clean-room compliant build for V12 VPU methodology project
REM Usage: build.bat [rebuild|run|clean]
REM Set OPENVINO_ROOT environment variable to your OpenVINO toolkit installation.

setlocal enabledelayedexpansion

REM --- Configuration ---
REM OPENVINO_ROOT: use environment variable if set
if not defined OPENVINO_ROOT (
    echo ERROR: OPENVINO_ROOT is not set.
    echo Set it to your OpenVINO toolkit path, e.g.:
    echo     set OPENVINO_ROOT=C:\path\to\openvino_toolkit
    exit /b 1
)

set "VS_TOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "PROJECT_ROOT=%~dp0"
set "BUILD_DIR=build"

REM --- Parse arguments ---
set "ACTION=build"
if /i "%~1"=="rebuild" set "ACTION=rebuild"
if /i "%~1"=="run" set "ACTION=run"
if /i "%~1"=="clean" set "ACTION=clean"

REM --- Initialize VS developer environment ---
call "%VS_TOOLS%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

REM --- Set up OpenVINO paths ---
set "PATH=%PATH%;%OPENVINO_ROOT%\runtime\bin\intel64\Release;%OPENVINO_ROOT%\runtime\3rdparty\tbb\bin"

cd /d "%PROJECT_ROOT%"

REM --- Clean action ---
if "%ACTION%"=="clean" (
    echo Cleaning build directory...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
    echo Clean complete.
    exit /b 0
)

REM --- Full rebuild ---
if "%ACTION%"=="rebuild" (
    echo Performing full rebuild...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
)

REM --- Configure ---
if not exist %BUILD_DIR% (
    echo Configuring CMake...
    cmake -S . -B %BUILD_DIR% -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: CMake configuration failed
        exit /b %ERRORLEVEL%
    )
)

REM --- Build ---
echo Building...
cmake --build %BUILD_DIR% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed
    exit /b %ERRORLEVEL%
)

echo.
echo Build successful.
echo.

REM --- Run ---
if "%ACTION%"=="run" (
    echo Running VPU MoE Lite...
    echo.
    %BUILD_DIR%\src\vpu_moe_lite.exe
    echo.
    echo Exit code: %ERRORLEVEL%
)

endlocal