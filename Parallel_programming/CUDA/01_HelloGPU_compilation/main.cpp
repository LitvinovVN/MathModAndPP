// Задача 01. Вывести с помощью GPU текстовую строку "Hello World from GPU!"
// Запуск:
// nvcc -O3 kernel.cu -dc -o target.o -gencode arch=compute_52,code=sm_52
// nvcc -O3 target.o -o dlink.o -gencode arch=compute_52,code=sm_52 -dlink
// g++ -c main.cpp -o main.o
// g++ dlink.o main.o target.o -o app -lcudadevrt -lcudart
// nvcc dlink.o main.o target.o -o app
// ./app

#include <iostream>
#include "kernel.cuh"

int main() {
    cuda_hello();
    return 0;
}