// Задача 01. Вывести с помощью GPU текстовую строку "Hello World from GPU!"
// Запуск:
// nvcc kernel.cu
// ./a

#include <iostream>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");    
}

int main() {
    cuda_hello<<<1,1>>>(); 
    return 0;
}