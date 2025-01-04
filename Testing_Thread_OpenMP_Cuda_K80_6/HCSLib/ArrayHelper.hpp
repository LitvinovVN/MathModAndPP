#pragma once

#include <iostream>

/// @brief Структура для хранения методов обработки массивов T*
struct ArrayHelper
{
    ////////////////////////// Суммирование элементов массива (начало) /////////////////////////////

    ///// Последовательное суммирование на CPU /////
    template<typename T>
    static T Sum(T* data, size_t indStart, size_t indEnd)
    {
        T result = 0;
        for (size_t i = indStart; i <= indEnd; i++)
        {
            result += data[i];
        }
        return result;
    }

    template<typename T>
    static T Sum(T* data, size_t size)
    {
        T result = Sum(data, 0, size-1);
        return result;
    }
    ///////////////////////////////////////////////

    ///// Суммирование с помощью std::thread на CPU //////
    // Структура для передачи аргументов в потоковую функцию
    template<typename T>
    struct SumThreadArgs
    {
        T* data;
        size_t indStart;
        size_t indEnd;
        T& sum;
        std::mutex& m;

        SumThreadArgs(T* data,
            size_t indStart,
            size_t indEnd,
            T& sum,
            std::mutex& m) : 
                data(data),
                indStart(indStart),
                indEnd(indEnd),
                sum(sum),
                m(m)
        {}
    };

    // Функция для исполнения потоком std::thread
    template<typename T>
    static void SumThread(SumThreadArgs<T> args)
    {
        T* data = args.data;
        auto indStart = args.indStart;
        auto indEnd = args.indEnd;
        T local_sum = 0;
        
        for (size_t i = indStart; i <= indEnd; i++)
        {
            local_sum += data[i];
        }
        
        {
            std::lock_guard<std::mutex> lock(args.m);
            args.sum += local_sum;
        }
    }

    template<typename T>
    static T Sum(T* data, size_t indStart, size_t indEnd, unsigned threadsNum)
    {
        std::mutex m;
        T sum = 0;
        size_t blockSize = indEnd - indStart + 1;
        std::vector<std::thread> threads;
        size_t thBlockSize = blockSize / threadsNum;
        
        for (size_t i = 0; i < threadsNum; i++)
        {
            size_t thIndStart = i * thBlockSize;
            size_t thIndEnd = thIndStart + thBlockSize - 1;
            if(i == threadsNum - 1)
                thIndEnd = indEnd;
                        
            SumThreadArgs<T> args(data, thIndStart, thIndEnd, sum, m);
            threads.push_back(std::thread(SumThread<T>, args));
        }
        
        for(auto& th : threads)
        {
            th.join();
        }

        return sum;
    }

    template<typename T>
    static T Sum(T* data, size_t size, unsigned threadsNum)
    {
        return Sum(data, 0, size - 1, threadsNum);
    }
    ///////////////////////////////////////////////

    ///// Суммирование с помошью OpenMP на CPU /////
    template<typename T>
    static T SumOpenMP(T* data, size_t indStart, size_t indEnd, unsigned threadsNum)
    {
        #ifdef _OPENMP
        omp_set_num_threads(threadsNum);
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (long long i = (long long)indStart; i <= (long long)indEnd; i++)
        {
            sum += data[i];
        }
        return sum;
        #else
            throw std::runtime_error("OpenMP not supported!");
        #endif
    }

    template<typename T>
    static T SumOpenMP(T* data, size_t size, unsigned threadsNum)
    {
        return SumOpenMP(data, 0, size - 1, threadsNum);
    }


    ///// Суммирование с помощью Cuda /////      

    template<typename T>
    static T SumCuda(T* dev_arr, size_t indStart, size_t indEnd, unsigned blocksNum, unsigned threadsNum)
    {
        #ifdef __NVCC__

        size_t length = indEnd - indStart + 1;
                        
        #ifdef DEBUG
        std::cout << "T Sum(" << dev_arr << ", "
                  << length << ", "<< blocksNum << ", "
                  << threadsNum << ")" <<std::endl;
        #endif
        
        T sum{0};
        T* dev_sum;
        cudaMalloc(&dev_sum, sizeof(T));
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
        kernel_sum<<<blocksNum, threadsNum, shared_mem_size>>>(dev_arr, length, dev_block_sum, dev_sum);

        //cudaMemcpy(&sum, dev_sum, sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum, dev_block_sum, blocksNum * sizeof(T), cudaMemcpyDeviceToHost);
        for(unsigned i=0; i<blocksNum;i++)
        {
            //std::cout << "block_sum[" << i << "] = " << block_sum[i] << std::endl;
            sum += block_sum[i];
        }

        #ifdef DEBUG
        std::cout << "Sum is " << sum << std::endl;
        #endif

        free(block_sum);
        cudaFree(dev_block_sum);

        return sum;

        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    template<typename T>
    static T SumCuda(T* data, size_t size, unsigned blocksNum, unsigned threadsNum)
    {
        return SumCuda(data, 0, size - 1, blocksNum, threadsNum);
    }

    ////////////////////////// Суммирование элементов массива (конец) /////////////////////////////

    /*  ---   Другие алгоритмы   ---  */

};

