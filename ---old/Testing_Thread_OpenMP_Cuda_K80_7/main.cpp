// set PATH=%PATH%;C:\mingw64\bin
// g++  main.cpp -o app -fopenmp -O3 -Wall
// g++  main.cpp -o app -lpthread -O3 -Wall
// --- nvcc, cl, windows ---
// nvcc main.cpp -o app -x cu -Xcompiler="/openmp -Wall"  -allow-unsupported-compiler -std=c++17
// nvcc main.cpp -o app -x cu -Xcompiler="/openmp" -allow-unsupported-compiler
// --- nvcc, g++, ubuntu ---
// nvcc main.cpp -o app -x cu -Xcompiler="-fopenmp -Wall" -O3
// nvcc main.cpp -o app -x cu -Xcompiler="-fopenmp" -O3

#include "HCSLib/_IncludeLib.hpp"


int main()
{
    std::cout << "Starting application..." << std::endl;
    Application app;
    app.Start();
}