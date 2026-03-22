1. opencv and MSYS2 MINGW64 to be downloaded
2. Open MSYS2 MINGW64 and navigate to the directory where "main.cpp" is present
3. run  g++ "main.cpp" -o "build/Debug/outDebug.exe" `pkg-config --cflags --libs opencv4`
4. Run "outDebug.exe"
