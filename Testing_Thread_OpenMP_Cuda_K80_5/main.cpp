// set PATH=%PATH%;C:\mingw64\bin
// g++  main.cpp -o app -fopenmp -O3 -Wall
// g++  main.cpp -o app -lpthread -O3 -Wall
// nvcc main.cpp -o app -Xcompiler="/openmp -Wall"  -x cu -allow-unsupported-compiler -std=c++17
// nvcc main.cpp -o app -Xcompiler="-fopenmp -Wall" -x cu

#include "HCSLib/_IncludeLib.hpp"


int main()
{
    std::cout << "Starting application..." << std::endl;
    Application app;
    app.Start();
}