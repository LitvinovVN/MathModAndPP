#pragma once

// cuda-ядро для вывода одномерного массива в консоль
template<typename T>
__global__
void print_kernel(T* data, size_t indStart, size_t length)
{
    int th_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_i == 0)
    {
        //printf("GPU: print_kernel() vectorGpu._size = %d\n", vectorGpu.GetSize());
        T* _dev_data_pointer = data;
        auto indEnd = indStart + length - 1;
        /*if(indEnd > vectorGpu.GetSize())
        {
            printf("Error! indEnd > vectorGpu.GetSize()\n");
            return;
        }*/

        printf("[%d..", (long)indStart);
        printf("%d]: ", (long)indEnd);
        for(size_t i = indStart; i <= indEnd; i++)
        {
            printf("%f ", _dev_data_pointer[i]);
        }
        printf("\n");
    }
}


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
    int n_elem_per_thread = length / number_of_threads;
        
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
    
}
