#pragma once

#include <iostream>

template<typename T>
__global__ void kernel_sum(T* dev_arr, size_t length, T* dev_block_sum, T* result)
{
    // Массив в распределенной памяти GPU
    // для хранения локальных сумм отдельных потоков блока
    extern __shared__ T shared_array[];

    //printf("\nkernel_sum: length = %ld\n", length);
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    //printf("\nkernel_sum: tid = %ld\n", tid);    
    const int number_of_threads = gridDim.x * blockDim.x;
    int n_elem_per_thread = length / (gridDim.x * blockDim.x);
        
    #ifdef DEBUG
    if(tid == 0)
    {
        printf("\nkernel_sum: number_of_threads = %d\n", number_of_threads);
        printf("\nkernel_sum: n_elem_per_thread = %d\n", n_elem_per_thread);
    }
    #endif

    unsigned long long block_start_idx = n_elem_per_thread * blockIdx.x * blockDim.x;
    unsigned long long thread_start_idx = block_start_idx
            + threadIdx.x * n_elem_per_thread;
    unsigned long long thread_end_idx = thread_start_idx + n_elem_per_thread;
    if(tid == number_of_threads - 1)
    {
        thread_end_idx = length;
    }

    if(thread_end_idx > length) thread_end_idx = length;
    
    #ifdef DEBUG
    printf("\nkernel_sum: i = %d [%d .. %d]\n", tid, thread_start_idx, thread_end_idx);
    #endif

    T localResult{0};
    
    for(size_t i = thread_start_idx; i < thread_end_idx; i++)
    {
        localResult += dev_arr[i];
    }

    #ifdef DEBUG    
    printf("\nkernel_sum: i = %d, localResult = %f\n", tid, localResult);
    #endif
    
    shared_array[threadIdx.x] = localResult;
    __syncthreads();

    // Просматриваем содержимое распределяемой памяти
    #ifdef DEBUG
    if(threadIdx.x == 0)
    {
        for(int i = 0; i < blockDim.x; i++)
        {
            printf("\nkernel_sum: %d (b%d, t%d) shared_array[%d] = %f\n", tid, blockIdx.x, threadIdx.x, i, shared_array[i]);
        }
    }
    #endif
    
    if(threadIdx.x == 0)
    {
        T block_result = 0;
        //atomicAdd(block_result, localResult);
        for(int i = 0; i < blockDim.x; i++)
        {
            block_result += shared_array[i];
            #ifdef DEBUG
            printf("\nkernel_sum: shared_array[%d] = %f\n", tid, shared_array[i]);
            #endif
        }
        #ifdef DEBUG
        printf("\nkernel_sum: %d, block_result = %f\n", tid, block_result);
        #endif
        dev_block_sum[blockIdx.x] = block_result;
    }
    
    /*__threadfence_system();
    
    // Поток с индексом 0 складывает все элементы массива dev_block_sum
    if(tid == 0)
    {
        
        T globalResult = 0;
        for(int i = 0; i < gridDim.x; i++)
        {
            globalResult += dev_block_sum[i];
            #ifdef DEBUG
            printf("\nkernel_sum: tid=%d, dev_block_sum[%d] = %f\n", tid, i, dev_block_sum[i]);
            #endif
        }
        *result = globalResult;

        #ifdef DEBUG
        printf("\nkernel_sum: *result = %f\n", *result);
        #endif
    }*/
}

template<typename T>
class CudaHelper
{
public:

    static void WriteGpuSpecs(std::ofstream& out)
    {
        out << "WriteGpuSpecs()" << std::endl;

        int nDevices;
        cudaGetDeviceCount(&nDevices);
        for (int i = 0; i < nDevices; i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            out << "Device Number: "             << i << std::endl;
            out << "  Device name: "             << prop.name << std::endl;
            out << "  Compute capability: "      << prop.major << "." << prop.minor << std::endl;
            out << "  MultiProcessorCount: "     << prop.multiProcessorCount << std::endl;
            out << "  asyncEngineCount: "        <<  prop.asyncEngineCount<< " (Number of asynchronous engines)" << std::endl;
            out << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
            out << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
            out << "  Peak Memory Bandwidth (GB/s): "
                << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
        }
    }

    static T Sum(T* dev_arr, size_t length, unsigned blocksNum, unsigned threadsNum)
    {
        #ifdef DEBUG
        std::cout << "T Sum(" << dev_arr << ", "
                  << length << ", "<< blocksNum << ", "
                  << threadsNum << ")" <<std::endl;
        #endif
        
        T result{0};
        T* dev_result;
        cudaMalloc(&dev_result, sizeof(T));
        //cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

        // Выделяем в распределяемой памяти каждого SM массив для хранения локальных сумм каждого потока блока
        unsigned shared_mem_size = threadsNum * sizeof(T);
        #ifdef DEBUG
        std::cout << "shared_mem_size = " << shared_mem_size << std::endl;
        #endif
        // Выделяем в RAM и глобальной памяти GPU массив для локальных сумм каждого блока
        T* block_sum = (T*)malloc(blocksNum * sizeof(T));
        T* dev_block_sum;
        cudaMalloc(&dev_block_sum, blocksNum * sizeof(T));
        kernel_sum<<<blocksNum, threadsNum, shared_mem_size>>>(dev_arr, length, dev_block_sum, dev_result);

        //cudaMemcpy(&result, dev_result, sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum, dev_block_sum, blocksNum * sizeof(T), cudaMemcpyDeviceToHost);
        for(int i=0; i<blocksNum;i++)
        {
            //std::cout << "block_sum[" << i << "] = " << block_sum[i] << std::endl;
            result += block_sum[i];
        }

        #ifdef DEBUG
        std::cout << "Sum is " << result << std::endl;
        #endif

        free(block_sum);
        cudaFree(dev_block_sum);

        return result;
    }
};
