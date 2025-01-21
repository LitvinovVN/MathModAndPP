#pragma once

#include <iostream>

/// @brief Структура для хранения методов обработки массивов T*
struct ArrayHelper
{
    ////////////////////////// Вывод массивов в консоль (начало) /////////////////////////////

    ///////// Вывод значений элементов массивов GPU в консоль
    template<typename T>
    static void PrintArrayCuda(T* data, size_t indStart, size_t length)
    {
        #ifdef __NVCC__
        
        kernel_print<T><<<1,1>>>(data, indStart, length);
        cudaDeviceSynchronize();

        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    ////////////////////////// Вывод массивов в консоль (конец) /////////////////////////////


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

    // Суммирование на одном GPU
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
        //T* dev_sum;
        //cudaMalloc(&dev_sum, sizeof(T));
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
        kernel_sum<<<blocksNum, threadsNum, shared_mem_size>>>(dev_arr, length, dev_block_sum);

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

    

    // Суммирование на нескольких GPU (Tesla K80)
    template<typename T>
    static T SumCudaMultiGpu(std::vector<ArrayGpuProcessingParams<T>> params)
    {
        std::cout << "SumCudaMultiGpu(std::vector<ArrayGpuProcessingParams<T>> params)\n\n";
        #ifdef __NVCC__
        
        T sum{0};
        
        auto gpuNum = params.size();

        std::vector<std::thread> threads;
        std::mutex mutex;
        for(int i = 0; i < gpuNum; i++)
        {
            threads.push_back(std::thread{[i, &mutex, &params, &sum]() {
                cudaSetDevice(i);
                T gpu_sum = SumCuda(params[i].dev_arr,
                                    params[i].indStart,
                                    params[i].indEnd,
                                    params[i].blocksNum,
                                    params[i].threadsNum );
                mutex.lock();
                std::cout << "thread " << i <<": ";
                params[i].Print();
                std::cout << "gpu_sum = " << gpu_sum <<"\n";
                sum += gpu_sum;
                mutex.unlock();
            }});
        }

        /*
        unsigned deviceId;
        T* dev_arr;
        size_t indStart;
        size_t indEnd;
        unsigned blocksNum;
        unsigned threadsNum;
        */

        for(auto& thread : threads)
        {
            thread.join();
        }

        return sum;

        /*#pragma omp parallel for num_threads(2)
        for(int i=0;i<2;i++)
        {
            cudaSetDevice(i);
            //kernel_sum<<<blocksNum, threadsNum, shared_mem_size>>>(dev_arr, length, dev_block_sum);
            if(i==0)
            {
                kernel_sum<<<blocksNum/2, threadsNum, shared_mem_size>>>(dev_arr, length/2, dev_block_sum);
            }
            else if(i==1)
            {
                kernel_sum<<<blocksNum - blocksNum/2, threadsNum, shared_mem_size>>>(dev_arr + length/2, length - length/2, dev_block_sum + blocksNum/2);
            }
                        
            //cudaMemcpy(...);
            //k_my_kernel<<<...>>>(...);
            //cudaMemcpy(...);
        }*/

        
        //cudaSetDevice(1); cudaDeviceSynchronize();
        //cudaSetDevice(0); cudaDeviceSynchronize();
               

        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }


    ////////////////////////// Суммирование элементов массива (конец) /////////////////////////////

    /*  ---   Другие алгоритмы   ---  */





    //////// 

    // Работа с функцией SumOpenMP
    static void SumOpenMP_ConsoleUI()
    {
        // Вызов функции суммирования с помощью OpenMP
        try
        {            
            size_t size  = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array size: ");
            double value = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");
            int Nthreads = ConsoleHelper::GetIntFromUser("Enter num threads: ");

            double* data = new double[size];
            for (size_t i = 0; i < size; i++)
            {
                data[i] = value;
            }            
            
            auto start = high_resolution_clock::now();
            double sum = ArrayHelper::SumOpenMP(data, 0, size, Nthreads);
            auto stop = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(stop - start);        
            auto t = duration.count();

            std::cout << "ArrayRamHelper::SumOpenMP(data, 0, size, Nthreads): " << sum << std::endl;
            std::cout << "Expected sum: " << size*value << std::endl;
            std::cout << "Time, mks: " << t << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;            
        }
    }

    /// @brief Выделяет память для массива на GPU
    /// @tparam T Тип элементов массива
    /// @param size Количество элементов
    /// @param deviceId Идентификатор устройства
    /// @return Указатель на созданный массив
    template<typename T>
    static T* CreateArrayGpu(unsigned long long size, int deviceId = 0)
    {
        #ifdef __NVCC__

        if (size == 0)
        {
            std::string mes = "Cannot initialize array of 0 elements";
            //std::cerr << mes << std::endl;
            throw std::logic_error(mes);
        }
        if(deviceId != 0)
            cudaSetDevice(deviceId);
        
        // Set curent CUDA device.
        cudaError_t cuda_status = cudaSetDevice(deviceId);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "Cannot set current CUDA device, status = %d: %s\n",
            cuda_status, cudaGetErrorString(cuda_status));
            throw std::runtime_error("Cannot set current CUDA device");
        }


        T* dev_array = nullptr;
        cudaError_t cudaResult = cudaMalloc(&dev_array, size*sizeof(T));
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not allocate device memory for GPU array: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }

        return dev_array;
        
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    template<typename T>
    static void InitArrayGpu(T* dev_array,
                        unsigned long long size,
                        T value,
                        int deviceId = 0)
    {
        #ifdef __NVCC__
        
        if(deviceId != 0)
            cudaSetDevice(deviceId);
        
        // Set curent CUDA device.
        cudaError_t cudaResult = cudaSetDevice(deviceId);
        if (cudaResult != cudaSuccess)
        {
            fprintf(stderr, "Cannot set current CUDA device, status = %d: %s\n",
            cudaResult, cudaGetErrorString(cudaResult));
            throw std::runtime_error("Cannot set current CUDA device");
        }
        
        //cudaError_t cudaResult = cudaMalloc(&dev_array, size*sizeof(T));
        kernel_array_init_by_value<<<1,1>>>(dev_array, 0, size, value);
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not init GPU array by value: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }
        
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }


    // Работа с функцией SumCudaMultiGpu(std::vector<ArrayGpuProcessingParams<T>> params)
    static void SumCudaMultiGpu_ConsoleUI()
    {
        std::cout << "SumCudaMultiGpu(std::vector<ArrayGpuProcessingParams<T>> params)\n";
        // Вызов функции суммирования с помощью Cuda на нескольких GPU
        try
        {
            int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();
            std::cout << "cudaDeviceNumber: " << cudaDeviceNumber << std::endl;
            double expectedResult = 0;
            std::vector<ArrayGpuProcessingParams<double>> params;
            for(int i = 0; i < cudaDeviceNumber; i++)
            {
                std::cout << "Init " << i << " array:\n";
                //size_t size    = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array size: ");
                //double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");
                //int blocksNum  = ConsoleHelper::GetIntFromUser("Enter num blocks: ");
                //int threadsNum = ConsoleHelper::GetIntFromUser("Enter num threads: ");
                size_t size    = (i+1)*10;
                double value   = i+0.1;
                int blocksNum  = 10;
                int threadsNum = 16;
                
                expectedResult += size*value;

                ArrayGpuProcessingParams<double> param;
                param.deviceId   = i;
                param.indStart   = 0;
                param.indEnd     = size-1;
                param.blocksNum  = blocksNum;
                param.threadsNum = threadsNum;
                try
                {
                    param.dev_arr = CreateArrayGpu<double>(size, i);
                    InitArrayGpu(param.dev_arr, size, value, i);
                    PrintArrayCuda(param.dev_arr, 0, size);
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                    std::exit(-1);
                }
                                
                params.push_back(param);
                params[i].Print();
            }
            
            auto start = high_resolution_clock::now();
            double sum = ArrayHelper::SumCudaMultiGpu(params);
            auto stop = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(stop - start);        
            auto t = duration.count();

            std::cout << "ArrayRamHelper::SumOpenMP(data, 0, size, Nthreads): " << sum << std::endl;
            std::cout << "Expected sum: " << expectedResult << std::endl;
            std::cout << "Time, mks: " << t << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;            
        }
    }

};

