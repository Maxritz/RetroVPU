@echo off
REM VPU MoE Lite - Run Script
REM Requires OPENVINO_ROOT environment variable and a prior build (build.bat).

if not defined OPENVINO_ROOT (
    echo ERROR: OPENVINO_ROOT is not set.
    echo Set it to your OpenVINO toolkit path.
    exit /b 1
)

set PATH=%PATH%;%OPENVINO_ROOT%\runtime\bin\intel64\Release;%OPENVINO_ROOT%\runtime\3rdparty\tbb\bin
cd /d %~dp0
build\src\vpu_moe_lite.exe
echo "EXIT_CODE=%ERRORLEVEL%"