# Use MSYS2 UCRT64

# One-time setup
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-pkgconf

# Go to your project folder
cd /e/Check
mkdir -p build/Debug

# Compile (use the refactored file)
g++ -std=c++17 main_refactored.cpp -o build/Debug/outDebug.exe $(pkg-config --cflags --libs opencv4)

# Run it
./build/Debug/outDebug.exe
