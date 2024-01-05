// Работает
// nvcc main.cu kernel.cu
// ./a

#include <iostream>
#include "kernel.cuh"

int main() {
    cuda_hello();
    return 0;
}