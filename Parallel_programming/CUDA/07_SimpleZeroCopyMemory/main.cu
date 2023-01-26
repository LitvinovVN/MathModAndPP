// Задача 07. Создать переменную counter = 0 в нуль-копируемой памяти.
// Изменять значения переменной counter в диапазоне от 1 до 5 с паузой 1 секунда в функции CPU.
// Выводить значения переменной counter, если она больше 0, с некоторой паузой в CUDA-ядре.
// Запуск:
// nvcc main.cu -o app
// ./app

#include <iostream>              // подключаем заголовочный файл iostream (содержит определение std::cout)
#include <thread>                // подключаем библиотеку для работы с потоками
#include <chrono>                // sleep_for
#include "cuda.h"

using namespace std::chrono_literals;// для использования единиц измерения времени (ms)

void thread_function(int* counter)                 
{
    std::cout << "Thread function: counter = " << *counter << std::endl;

    while(*counter < 5)
    {
        (*counter)++;        
        std::cout << "Thread function: counter = " << *counter << std::endl;
        std::this_thread::sleep_for(1000ms);
    }    
}


using clock_value_t = long long;

__device__ void cuda_sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < sleep_cycles);
}

__global__
void cuda_kernel_function(int* dev_counter)
{
    printf("cuda_kernel_function started\n");

    while(true)    
    {
        if(*dev_counter < 1) continue;

        printf("dev_counter = %d\n", *dev_counter);
        cuda_sleep(300000000);

        if(*dev_counter >= 5) break;
    }

    printf("cuda_kernel_function stopped\n");
}

int main()
{
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    int *counter = NULL;
    cudaHostAlloc((void**)&counter, sizeof(int), cudaHostAllocMapped);

    int *dev_counter = NULL;
    cudaHostGetDevicePointer(&dev_counter, counter, 0);

    std::cout << "Main thread: Starting new thread...\n";

    cudaEventRecord(cuda_start, 0);
    auto start = std::chrono::high_resolution_clock::now();

    cuda_kernel_function<<<1,1>>>(dev_counter);    // cuda_kernel_function starts running

    std::thread t(&thread_function, counter);   // t starts running
    std::cout << "Main thread: New thread started!\n";

    cudaDeviceSynchronize();
    t.join();   // main thread waits for the thread t to finish
    
    std::cout << "Main thread: Thread joined\n";

    auto end = std::chrono::high_resolution_clock::now();
    cudaEventRecord(cuda_stop, 0);

    std::chrono::duration<double, std::milli> elapsed = end-start;
    cudaEventSynchronize(cuda_start);
    cudaEventSynchronize(cuda_stop);
    float cuda_elapsedTime;
    cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);

    std::cout << "---------------" << std::endl;
    std::cout << "Waited " << elapsed.count() << " ms (std::chrono)\n";
    std::cout << "Waited " << cuda_elapsedTime << " ms (CUDA)\n";
    
    std::cout << "counter = " << *counter << std::endl;

    return 0;
}