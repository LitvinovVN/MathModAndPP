// Задача 14. Реализовать в управляющем потоке механизм ожидания статусов READE для всех вычислительных потоков.
// Процесс конвейерного вычисления
// с одним управляющим и двумя рабочими потоками CPU и с тремя блоками на GPU.
// У каждого потока и блока свой id
// Номер шага конвейера r изменяется управляющим потоком CPU в диапазоне от 1 до 15 с паузой в 1 секунду.
// Каждый поток циклически с некоторой паузой выводит свой id и текущее значение r.
// Реализовать динамическое выделение распределяемой памяти
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

enum ThreadStatus { INIT = 0, BUSY = 1, READY = 2 };

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
// int threadIndexGlobal - глобальный индекс вычислительного потока
// int* counter - счетчик шага конвейерного вычисления
// int* thStatArray - указатель на массив статусов вычислительных потоков в нуль-копируемой памяти
void thread_function_work(int threadIndexGlobal, int* counter, int* thStatArray)
{
    thStatArray[threadIndexGlobal] = INIT;
    PrintThread{} << "thread_function_work: " << threadIndexGlobal << " | th_status = " << thStatArray[threadIndexGlobal] << std::endl;
            
    while(true)
    {
        if(thStatArray[threadIndexGlobal] != READY)
            thStatArray[threadIndexGlobal] = READY;

        if(*counter > NUM_R) break;
        
        if(*counter < 1) continue; // На шаге 0 никто не считает
        if(threadIndexGlobal + 1 > *counter) continue; // Разгон конвейера
        if(*counter > (NUM_R - GRID_BLOCK_DIM + threadIndexGlobal + 1)) break; // Останов конвейера

        thStatArray[threadIndexGlobal] = BUSY;
        //PrintThread{} << "thread_function_work: " << threadIndexGlobal << " | counter = " << *counter << " | th_status = " << thStatArray[threadIndexGlobal] << std::endl;
        
        std::this_thread::sleep_for(200ms);// Имитация занятости потока

        thStatArray[threadIndexGlobal] = READY;
        PrintThread{} << "thread_function_work: " << threadIndexGlobal << " | counter = " << *counter << " | th_status = " << thStatArray[threadIndexGlobal] << std::endl;
    }
    PrintThread{} << "thread_function_work ended: " << threadIndexGlobal << " | counter = " << *counter << std::endl;
}

// Функция управляющего потока CPU
// int* counter - указатель на счетчик шагов конвейера в нуль-копируемой памяти
// int* thStatArray - указатель на массив статусов вычислительных потоков в нуль-копируемой памяти
// int thStatArrayLength - количество элементов массива thStatArray
void thread_function(int* counter, int* thStatArray, int thStatArrayLength)                 
{    
    while(*counter < NUM_R+1)
    {                
        PrintThread{} << "Controller thread function: counter = " << *counter << std::endl;

        if(*counter > 0) std::this_thread::sleep_for(1000ms);

        // Ожидание состояния READY для всех вычислительных потоков        
        /*while(true)
        {
            bool isREADY = true;
            for(int i = 0; i < thStatArrayLength; i++)
            {
                std::this_thread::sleep_for(100ms);
                if (thStatArray[i]!=READY) isREADY = false;
            }
            

            PrintThread{} << "Controller thread function: calc threads statuses = [";
            for(int i = 0; i < thStatArrayLength; i++)
            {
                PrintThread{} << thStatArray[i] << " ";
            }
            PrintThread{} << "]" << std::endl;

            if (isREADY) break;
        }*/

        PrintThread{} << "Controller thread function: calc threads statuses = [";
        for(int i = 0; i < thStatArrayLength; i++)
        {
            PrintThread{} << thStatArray[i] << " ";
        }
        PrintThread{} << "]" << std::endl;

        
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

// Функция рабочего потока GPU (cuda-ядро)
// int device_threads_offset - глобальное смещение первого вычислительного блока cuda (для вычисления глобального индекса потока)
// int* dev_counter - счетчик шага конвейерного вычисления
// int* dev_thStatArray - указатель на массив статусов вычислительных потоков в нуль-копируемой памяти
__global__
void cuda_kernel_function(int device_threads_offset, int* dev_counter, int* dev_thStatArray)
{    
    printf("cuda_kernel_function started\n");
    int blockIndex = blockIdx.x;
    int threadIndexGlobal = device_threads_offset + blockIndex;

    int threadIndexLocal = threadIdx.x;
    if (threadIndexLocal == 0) dev_thStatArray[threadIndexGlobal] = INIT;

    //__shared__ int shm_counter[BLOCK_DIM];
    extern __shared__ int shm_counter[];
    

    while(true)    
    {
        __syncthreads();
        if (threadIndexLocal == 0) dev_thStatArray[threadIndexGlobal] = READY;

        if(threadIndexLocal == 0) shm_counter[blockIndex] = *dev_counter;
        int tmp_counter = shm_counter[blockIndex];
        
        if(tmp_counter > NUM_R) break;
        
        if(tmp_counter < 1) continue; // На шаге 0 никто не считает
        if(threadIndexGlobal + 1 > tmp_counter) continue; // Разгон конвейера
        if(tmp_counter > (NUM_R - GRID_BLOCK_DIM + threadIndexGlobal + 1)) break; // Останов конвейера

        if (threadIndexLocal == 0) dev_thStatArray[threadIndexGlobal] = BUSY;
        printf("threadIndexGlobal = %d, blockIdx.x = %d, dev_counter = %d, thread block status = %d\n", threadIndexGlobal, blockIndex, tmp_counter, dev_thStatArray[threadIndexGlobal]);
        cuda_sleep(300000000);
        if (threadIndexLocal == 0) dev_thStatArray[threadIndexGlobal] = READY;
        printf("threadIndexGlobal = %d, blockIdx.x = %d, dev_counter = %d, thread block status = %d\n", threadIndexGlobal, blockIndex, tmp_counter, dev_thStatArray[threadIndexGlobal]);
    }

    printf("cuda_kernel_function stopped\n");
}

int main()
{
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    // Счетчик шагов конвейера
    int *counter = NULL;
    cudaHostAlloc((void**)&counter, sizeof(int), cudaHostAllocMapped);
    int *dev_counter = NULL;
    cudaHostGetDevicePointer(&dev_counter, counter, 0);

    // Массив статусов вычислительных потоков
    int numThreads = 5; // 2 CPU + 3 GPU
    int *thStatArray = NULL;
    cudaHostAlloc((void**)&thStatArray, numThreads * sizeof(int), cudaHostAllocMapped);
    int *dev_thStatArray = NULL;
    cudaHostGetDevicePointer(&dev_thStatArray, thStatArray, 0);

    std::cout << "Main thread: Starting new thread...\n";

    cudaEventRecord(cuda_start, 0);
    auto start = std::chrono::high_resolution_clock::now();

    //cuda_kernel_function<<<BLOCK_DIM,1>>>(dev_counter);    // cuda_kernel_function starts running
    int device_threads_offset = 2;
    size_t sharedMemoryValue = BLOCK_DIM*sizeof(int);// Резервируем разделяемую память под хранение счетчика шагов
    //sharedMemoryValue += BLOCK_DIM*sizeof(int);// Резервируем разделяемую память под хранение статуса вычислительного потока
    cuda_kernel_function<<<BLOCK_DIM,1,sharedMemoryValue>>>(device_threads_offset, dev_counter, dev_thStatArray);    // cuda_kernel_function starts running, динамическое выделение распределяемой памяти

    // ----- Создаём рабочие потоки CPU ----- 
    int th_id_0 = 0;// УИД первого рабочего потока CPU
    std::thread t_w0(&thread_function_work, th_id_0, counter, thStatArray);

    int th_id_1 = 1;// УИД второго рабочего потока CPU
    std::thread t_w1(&thread_function_work, th_id_1, counter, thStatArray);

    // ----- Запускаем управляющий поток -----
    std::thread t(&thread_function, counter, thStatArray, numThreads);   // t starts running
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