// Задача 11. Организовать процесс конвейерного вычисления
// с одним управляющим и двумя рабочими потоками CPU и с тремя блоками на GPU.
// У каждого потока и блока свой id
// Номер шага конвейера r изменяется управляющим потоком CPU в диапазоне от 1 до 15 с паузой в 1 секунду.
// Каждый поток циклически с некоторой паузой выводит свой id и текущее значение r.
// Реализовать динамическое выделение распределяемой памяти
// Источник: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
// Запуск:
// nvcc main.cu -o app
// ./app

#include <iostream>              // подключаем заголовочный файл iostream (содержит определение std::cout)
#include <thread>                // подключаем библиотеку для работы с потоками
#include <chrono>                // sleep_for
#include "cuda.h"
#include <mutex>
#include <sstream>

using namespace std::chrono_literals;// для использования единиц измерения времени (ms)

#define GRID_BLOCK_DIM 5 // Количество блоков расчетной сетки
#define BLOCK_DIM 3 // Количество потоковых блоков cuda
#define NUM_R 15

/** Thread safe cout class
  * Exemple of use:
  *    PrintThread{} << "Hello world!" << std::endl;
  */
class PrintThread: public std::ostringstream
{
public:
    PrintThread() = default;
  
    ~PrintThread()
    {
        std::lock_guard<std::mutex> guard(_mutexPrint);
        std::cout << this->str();
    }
  
private:
    static std::mutex _mutexPrint;
};
  
  std::mutex PrintThread::_mutexPrint{};

// Функция рабочего потока CPU
void thread_function_work(int threadIndexGlobal, int* counter)
{    
    while(true)
    {
        if(*counter > NUM_R) break;
        
        if(*counter < 1) continue; // На шаге 0 никто не считает
        if(threadIndexGlobal + 1 > *counter) continue; // Разгон конвейера
        if(*counter > (NUM_R - GRID_BLOCK_DIM + threadIndexGlobal + 1)) break; // Останов конвейера

        PrintThread{} << "thread_function_work: " << threadIndexGlobal << " | counter = " << *counter << std::endl;
        std::this_thread::sleep_for(200ms);
    }
    PrintThread{} << "thread_function_work ended: " << threadIndexGlobal << " | counter = " << *counter << std::endl;
}

// Функция управляющего потока CPU
void thread_function(int* counter)                 
{    
    while(*counter < NUM_R+1)
    {                
        PrintThread{} << "Controller thread function: counter = " << *counter << std::endl;
        if(*counter > 0) std::this_thread::sleep_for(1000ms);
        (*counter)++;
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
void cuda_kernel_function(int device_threads_offset, int* dev_counter)
{
    printf("cuda_kernel_function started\n");
    int blockIndex = blockIdx.x;
    int threadIndexGlobal = device_threads_offset + blockIndex;
    //__shared__ int shm_counter[BLOCK_DIM];
    extern __shared__ int shm_counter[];

    while(true)    
    {
        __syncthreads();
        if(threadIdx.x == 0) shm_counter[blockIndex] = *dev_counter;
        int tmp_counter = shm_counter[blockIndex];
        
        if(tmp_counter > NUM_R) break;
        
        if(tmp_counter < 1) continue; // На шаге 0 никто не считает
        if(threadIndexGlobal + 1 > tmp_counter) continue; // Разгон конвейера
        if(tmp_counter > (NUM_R - GRID_BLOCK_DIM + threadIndexGlobal + 1)) break; // Останов конвейера

        printf("threadIndexGlobal = %d, blockIdx.x = %d, dev_counter = %d\n", threadIndexGlobal, blockIndex, tmp_counter);
        cuda_sleep(300000000); 
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

    //cuda_kernel_function<<<BLOCK_DIM,1>>>(dev_counter);    // cuda_kernel_function starts running
    int device_threads_offset = 2;
    cuda_kernel_function<<<BLOCK_DIM,1,BLOCK_DIM*sizeof(int)>>>(device_threads_offset, dev_counter);    // cuda_kernel_function starts running, динамическое выделение распределяемой памяти

    // ----- Создаём рабочие потоки CPU ----- 
    int th_id_0 = 0;// УИД первого рабочего потока CPU
    std::thread t_w0(&thread_function_work, th_id_0, counter);

    int th_id_1 = 1;// УИД второго рабочего потока CPU
    std::thread t_w1(&thread_function_work, th_id_1, counter);

    // ----- Запускаем управляющий поток -----
    std::thread t(&thread_function, counter);   // t starts running
    std::cout << "Main thread: New thread started!\n";

    cudaDeviceSynchronize();

    t_w0.join();
    t_w1.join();

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