@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Users/rina0423/Desktop/openvino_toolkit/runtime/bin/intel64/Debug;C:\Users/rina0423/Desktop/openvino_toolkit/runtime/3rdparty/tbb/bin;%PATH%
test_minimal.exe
echo Exit: %ERRORLEVEL%
