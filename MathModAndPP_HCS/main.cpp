// sudo apt-get install libopenblas-dev
// set PATH=%PATH%;C:\mingw64\bin
// g++  main.cpp -o app -fopenmp -O3 -Wall
// g++  main.cpp -o app -lpthread -O3 -Wall
// g++  main.cpp -o app -fopenmp -O3 -Wall -DOPENBLAS -lopenblas -I"C:\OpenBLAS\include" -L"C:\OpenBLAS\lib"
// --- nvcc, cl, windows ---
// nvcc main.cpp -o app -x cu -Xcompiler="/openmp" -O3  -lcublas -allow-unsupported-compiler
// --- nvcc, cl, OpenBlas, windows ---
// nvcc main.cpp -o app -x cu -Xcompiler="/openmp" -O3  -lcublas -allow-unsupported-compiler -DOPENBLAS -lopenblas -I"C:\OpenBLAS\include" -L"C:\OpenBLAS\lib"
// --- nvcc, g++, ubuntu ---
// nvcc main.cpp -o app -x cu -Xcompiler="-fopenmp" -O3 -lcublas
// --- Код после препроцессинга ---
// g++ -E main.cpp -o main.ii

#include "HCSLib/_IncludeLib.hpp"

int main()
{    
    std::cout << "Starting application..." << std::endl;
    Application app;
    app.Start();
}