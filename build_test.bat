@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Users/rina0423/Desktop/openvino_toolkit/runtime/bin/intel64/Debug;%PATH%
cl /EHsc /std:c++17 /MDd /I "C:\Users/rina0423/Desktop/openvino_toolkit/runtime/include" test_minimal.cpp /link /LIBPATH:"C:\Users/rina0423/Desktop/openvino_toolkit/runtime/lib/intel64/Release" openvino.lib /OUT:test_minimal.exe 2>&1
if exist test_minimal.exe (
    echo Running...
    test_minimal.exe 2>&1
    echo Done. Exit: %ERRORLEVEL%
)
