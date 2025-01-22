#pragma once

#include <iostream>

/// @brief Структура для хранения методов обработки массивов T*
struct ArrayHelper
{
    ////////////////////////// Вывод массивов в консоль (начало) /////////////////////////////
    
    template<typename T>
    static void PrintArrayRam(T* data, size_t indStart, size_t length)
    {
        std::cout << "[";
        for (size_t i = indStart; i < indStart+length-1; i++)
        {
            std::cout << data[i] << " ";
        }
        std::cout << data[indStart+length-1];
        
        std::cout << "]\n";
    }


    ///////// Вывод значений элементов массивов GPU в консоль    
    template<typename T>
    static void PrintArrayGpu(T* data, size_t indStart, size_t length, int deviceId = 0)
    {
        #ifdef __NVCC__
        
        if(deviceId > 0)
        {
            std::thread th{[&](){
                cudaSetDevice(deviceId);
                kernel_print<T><<<1,1>>>(data, indStart, length);
                cudaDeviceSynchronize();
            }};
            th.join();
        }
        else
        {
            kernel_print<T><<<1,1>>>(data, indStart, length);
            cudaDeviceSynchronize();
        }

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
        std::cout << "SumCuda: Sum is " << sum << std::endl;
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
        //std::cout << "SumCudaMultiGpu(std::vector<ArrayGpuProcessingParams<T>> params)\n\n";
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
                //std::cout << "thread " << i <<": ";
                //params[i].Print();
                //std::cout << "gpu_sum = " << gpu_sum <<"\n";
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


    /// @brief Выделяет память для массива на текущем GPU
    /// @tparam T Тип элементов массива
    /// @param size Количество элементов
    /// @return Указатель на созданный массив
    template<typename T>
    static T* CreateArrayGpu(unsigned long long size)
    {
        #ifdef __NVCC__

        if (size == 0)
        {
            std::string mes = "Cannot initialize array of 0 elements";
            //std::cerr << mes << std::endl;
            throw std::logic_error(mes);
        }
        
        T* dev_array = nullptr;
        cudaMalloc(&dev_array, size*sizeof(T));
        
        std::string msg("Could not allocate device memory for GPU array: ");
        msg += std::to_string(size*sizeof(T));
        msg += " bytes not allocated!\n";
        cudaCheckErrors(msg.c_str());
        
        cudaDeviceSynchronize();

        return dev_array;
        
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }


    /// @brief Выделяет память для массива на GPU
    /// @tparam T Тип элементов массива
    /// @param size Количество элементов
    /// @param deviceId Идентификатор устройства
    /// @return Указатель на созданный массив
    template<typename T>
    static T* CreateArrayGpu(unsigned long long size, int deviceId)
    {
        #ifdef __NVCC__

        if (size == 0)
        {
            std::string mes = "Cannot initialize array of 0 elements";
            //std::cerr << mes << std::endl;
            throw std::logic_error(mes);
        }

        T* dev_array = nullptr;

        if(deviceId == 0)
        {
            dev_array = CreateArrayGpu<T>(size);
        }
        else
        {
            std::thread th{
                [&](){
                    // Set CUDA device.
                    cudaSetDevice(deviceId);                    
                    cudaCheckErrors("Cannot set CUDA device\n");
                    
                    dev_array = CreateArrayGpu<T>(size);
                }
            };
            th.join();
        }
                  
        

        return dev_array;
        
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    /// @brief Заполняет массив dev_array на текущем GPU значением value
    /// @tparam T Тип элементов массива
    /// @param dev_array Указатель на инициализируемый массив
    /// @param size Количество элементов массива
    /// @param value Значение, присваиваемое всем элементам массива dev_array
    template<typename T>
    static void InitArrayGpu(T* dev_array,
                        unsigned long long size,
                        T value)
    {
        #ifdef __NVCC__                       
        
        const unsigned blockSize = 100000000;
        if(size < blockSize)
        {
            kernel_array_init_by_value<<<1,1>>>(dev_array, 0, size, value);
            cudaError_t cudaResult = cudaGetLastError();
            if (cudaResult != cudaSuccess)
            {
                std::string msg("Could not init GPU array by value: ");
                msg += cudaGetErrorString(cudaResult);
                throw std::runtime_error(msg);
            }
            cudaDeviceSynchronize();
        }
        else
        {
            T* dev_array_ptr = dev_array;
            unsigned blocksNum = size / blockSize;
            for(unsigned blockIndex = 0; blockIndex < blocksNum; blockIndex++)
            {
                std::cout << "Init block " << blockIndex << std::endl;

                kernel_array_init_by_value<<<1,1>>>(dev_array_ptr, 0, blockSize, value);
                cudaError_t cudaResult = cudaGetLastError();
                if (cudaResult != cudaSuccess)
                {
                    std::string msg("Could not init GPU array by value: ");
                    msg += cudaGetErrorString(cudaResult);
                    throw std::runtime_error(msg);
                }
                cudaDeviceSynchronize();

                dev_array_ptr += blockSize;
            }

            unsigned lastBlockSize = size % blockSize;
            if(lastBlockSize > 0)
            {
                std::cout << "Init last block " << lastBlockSize << " elements" << std::endl;

                kernel_array_init_by_value<<<1,1>>>(dev_array_ptr, 0, lastBlockSize, value);
                cudaError_t cudaResult = cudaGetLastError();
                if (cudaResult != cudaSuccess)
                {
                    std::string msg("Could not init GPU array by value: ");
                    msg += cudaGetErrorString(cudaResult);
                    throw std::runtime_error(msg);
                }
                cudaDeviceSynchronize();
            }

        }
        
        
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    template<typename T>
    static void InitArrayGpu(T* dev_array,
                        unsigned long long size,
                        T value,
                        int deviceId)
    {
        #ifdef __NVCC__
        
        if(deviceId == 0)
        {
            InitArrayGpu(dev_array, size, value);
        }
        else
        {
            std::thread th{
                [&](){
                    // Set curent CUDA device.
                    cudaError_t cudaResult = cudaSetDevice(deviceId);
                    if (cudaResult != cudaSuccess)
                    {
                        fprintf(stderr, "Cannot set current CUDA device, status = %d: %s\n",
                        cudaResult, cudaGetErrorString(cudaResult));
                        throw std::runtime_error("Cannot set current CUDA device");
                    }
                    InitArrayGpu(dev_array, size, value);
                }
            };
            th.join();
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
            //cudaDeviceNumber = 1;
            std::cout << "cudaDeviceNumber: " << cudaDeviceNumber << std::endl;
            double expectedResult = 0;
            std::vector<ArrayGpuProcessingParams<double>> params;
            for(int i = 0; i < cudaDeviceNumber; i++)
            {
                std::cout << "--- Init " << i << " array starting...\n";
                //size_t size    = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array size: ");
                //double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");
                //int blocksNum  = ConsoleHelper::GetIntFromUser("Enter num blocks: ");
                //int threadsNum = ConsoleHelper::GetIntFromUser("Enter num threads: ");
                size_t size    = 500000000ull;
                double value   = 0.001;
                int blocksNum  = 34;
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
                    std::cout << "array " << i << " created\n";
                    std::cout << "First 10 elements of " << i << " array: ";                
                    PrintArrayGpu(param.dev_arr, 0, 10, i);

                    InitArrayGpu(param.dev_arr, size, value, i);
                    std::cout << "array " << i << " initialized\n";
                    std::cout << "First 10 elements of " << i << " array: ";                
                    PrintArrayGpu(param.dev_arr, 0, 10, i);
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                    std::exit(-1);
                }
                                
                params.push_back(param);
                params[i].Print();
                std::cout << "--- Initializing " << i << " array completed!\n";
            }
            
            auto start = high_resolution_clock::now();
            double sum = ArrayHelper::SumCudaMultiGpu(params);
            auto stop = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(stop - start);        
            auto t = duration.count();

            std::cout << "ArrayRamHelper::SumCudaMultiGpu(...): " << sum << std::endl;
            std::cout << "Expected sum: " << expectedResult << std::endl;
            std::cout << "Time, mks: " << t << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;            
        }
    }


    template<typename T>
    static void CopyRamToGpu(T* arrayRam, T* arrayGpu, size_t length)
    {
        #ifdef __NVCC__

        size_t dataSize = length * sizeof(T);
        cudaMemcpy(arrayGpu, arrayRam, dataSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaCheckErrors("Error in cudaMemcpy()");

        #else
        throw std::runtime_error("CUDA not supported!");
        #endif        
    }

    
    template<typename T>
    static void CopyRamToGpu(T* arrayRam, T* arrayGpu,
        size_t ind_start, size_t length, int deviceId = 0)
    {
        #ifdef __NVCC__

        if(deviceId == 0)
        {
            CopyRamToGpu(arrayRam + ind_start, arrayGpu + ind_start, length);
        }
        else
        {
            std::thread th{
                [&]() {
                    cudaSetDevice(deviceId);
                    CopyRamToGpu(arrayRam + ind_start, arrayGpu + ind_start, length);
                }
            };
            th.join();
        }

        #else
        throw std::runtime_error("CUDA not supported!");
        #endif
    }

    template<typename T>
    static void CopyGpuToRam(T* arrayGpu, T* arrayRam, size_t length)
    {
        #ifdef __NVCC__

        size_t dataSize = length * sizeof(T);
        cudaMemcpy(arrayRam, arrayGpu, dataSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaCheckErrors("Error in cudaMemcpy()");

        #else
        throw std::runtime_error("CUDA not supported!");
        #endif
    }

    
    template<typename T>
    static void CopyGpuToRam(T* arrayGpu, T* arrayRam,
        size_t ind_start, size_t length, int deviceId = 0)
    {
        #ifdef __NVCC__
        if(deviceId == 0)
        {
            CopyGpuToRam(arrayGpu + ind_start, arrayRam + ind_start, length);
        }
        else
        {
            std::thread th{
                [&]() {
                    cudaSetDevice(deviceId);
                    CopyGpuToRam(arrayGpu + ind_start, arrayRam + ind_start, length);
                }
            };
            th.join();
        }
        #else
        throw std::runtime_error("CUDA not supported!");
        #endif
    }

    template<typename T>
    static bool IsEqualsRamRam(T* arrayRam1, T*arrayRam2,
        size_t length, double eps = 0.00000001)
    {
        for (size_t i = 0; i < length; i++)
        {
            if(fabs(arrayRam2[i] - arrayRam1[i]) > eps)
                return false;
        }
        return true;
    }

    /// @brief Сравнивает содержимое массивов, расположенных на RAM и GPU
    /// @tparam T 
    /// @param arrayRam 
    /// @param arrayGpu 
    /// @param length 
    /// @param eps 
    /// @return 
    template<typename T>
    static bool IsEqualsRamGpu(T* arrayRam, T* arrayGpu,
        size_t length, double eps = 0.00000001)
    {
        #ifdef __NVCC__

        bool isEquals = true;
        const unsigned blockSize = 1000000;

        unsigned blocksNum     = length / blockSize;
        unsigned lastBlockSize = length % blockSize;
        
        T* arrayRamTmp = new T[blockSize];

        for (size_t blockInd = 0; blockInd < blocksNum; blockInd++)
        {
            CopyGpuToRam(arrayGpu+blockInd*blockSize, arrayRamTmp, blockSize);
            isEquals = IsEqualsRamRam(arrayRam+blockInd*blockSize, arrayRamTmp, blockSize);
            if(!isEquals) break;            
        }

        if(isEquals && lastBlockSize > 0)
        {
            CopyGpuToRam(arrayGpu+blocksNum*blockSize, arrayRamTmp, lastBlockSize);
            isEquals = IsEqualsRamRam(arrayRam+blocksNum*blockSize, arrayRamTmp, lastBlockSize);
        }

        delete[] arrayRamTmp;
        return isEquals;

        #else
        throw std::runtime_error("CUDA not supported!");
        #endif
    }
    
    /// @brief Копирование данных из RAM в GPU
    static void CopyRamToGpu_ConsoleUI()
    {
        std::cout << "CopyRamToGpu_ConsoleUI\n";
        try
        {
            int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();            
            std::cout << "cudaDeviceNumber: " << cudaDeviceNumber << std::endl;
            
            size_t length    = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length: ");
            double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");

            // Инициализируем массив в RAM
            double* arrayRam = new double[length];
            for (size_t i = 0; i < length; i++)
            {
                arrayRam[i] = value+0.1*i;
            }
            std::cout << "arrayRam[0]: " << arrayRam[0] << std::endl;
            std::cout << "arrayRam[length-1]: " << arrayRam[length-1] << std::endl;
                        
            for(int i = 0; i < cudaDeviceNumber; i++)
            {
                std::cout << "--- Starting work with GPU " << i << "  ---\n";

                std::cout << "Creating array on GPU " << i << "... ";
                double* arrayGpu = CreateArrayGpu<double>(length, i);
                std::cout << "OK\n";

                std::cout << "--- Copy to GPU" << i << " starting...\n";            
                auto start = high_resolution_clock::now();
                ArrayHelper::CopyRamToGpu(arrayRam, arrayGpu, 0, length, i);
                auto stop = high_resolution_clock::now();

                auto duration = duration_cast<microseconds>(stop - start);        
                auto t = duration.count();                
                std::cout << "Time, mks: " << t << std::endl;

                std::cout << "Ram: ";
                if(length>20)
                    PrintArrayRam(arrayRam, 0, 20);
                else
                    PrintArrayRam(arrayRam, 0, length);

                std::cout << "GPU " << i << ": ";                
                if(length>20)                
                    PrintArrayGpu(arrayGpu, 0, 20, i);
                else
                    PrintArrayGpu(arrayGpu, 0, length, i);

                //arrayRam[length-1]+=0.00001;
                bool isEquals = IsEqualsRamGpu(arrayRam, arrayGpu, length);
                if (isEquals)
                    std::cout << "Success! Arrays are equals!\n";
                else
                    std::cout << "Error! Arrays are not equals!\n";
                
                std::cout << "--------------------------------\n";
            }

            delete[] arrayRam;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;            
        }
    }

    /// @brief Копирование данных из GPU в RAM
    static void CopyGpuToRam_ConsoleUI()
    {
        std::cout << "CopyGpuToRam_ConsoleUI\n";
        try
        {
            int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();            
            std::cout << "cudaDeviceNumber: " << cudaDeviceNumber << std::endl;
            
            size_t size    = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array size: ");
            double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");

            // Инициализируем массив в RAM
            double* arrayRam = new double[size];
            for (size_t i = 0; i < size; i++)
            {
                arrayRam[i] = value;
            }
            //std::cout << "arrayRam[0]: " << arrayRam[0] << std::endl;
            //std::cout << "arrayRam[size-1]: " << arrayRam[size-1] << std::endl;
                        
            for(int i = 0; i < cudaDeviceNumber; i++)
            {
                std::cout << "--- Starting work with GPU " << i << "  ---\n";

                std::cout << "Creating array on GPU " << i << "... ";
                double* dev_array = CreateArrayGpu<double>(size, i);
                std::cout << "OK\n";

                std::cout << "Copy from RAM to GPU" << i << " starting...";
                {
                    auto start = high_resolution_clock::now();
                    ArrayHelper::CopyRamToGpu(arrayRam, dev_array, 0, size, i);
                    auto stop = high_resolution_clock::now();
                    std::cout << "OK\n";

                    auto duration = duration_cast<microseconds>(stop - start);        
                    auto t = duration.count();
                    std::cout << "Time, mks: " << t << std::endl;
                }

                std::cout << "Copy from GPU " << i << " to Ram starting...";
                double* arrayRamTmp = new double[size];
                
                {
                    auto start = high_resolution_clock::now();
                    ArrayHelper::CopyGpuToRam(dev_array, arrayRamTmp, 0, size, i);
                    auto stop = high_resolution_clock::now();                
                    std::cout << "OK\n";
                    
                    auto duration = duration_cast<microseconds>(stop - start);        
                    auto t = duration.count();                
                    std::cout << "Time, mks: " << t << std::endl;
                }

                std::cout << "Ram: ";
                if(size>20)
                    PrintArrayRam(arrayRam, 0, 20);
                else
                    PrintArrayRam(arrayRam, 0, size);

                std::cout << "GPU " << i << ": ";                
                if(size>20)                
                    PrintArrayGpu(dev_array, 0, 20, i);
                else
                    PrintArrayGpu(dev_array, 0, size, i);
                
                std::cout << "Ram copied: ";
                if(size>20)
                    PrintArrayRam(arrayRamTmp, 0, 20);
                else
                    PrintArrayRam(arrayRamTmp, 0, size);
                
                bool isEquals = IsEqualsRamRam(arrayRam, arrayRamTmp, size);
                if(isEquals)
                    std::cout << "Checking equals: OK\n";
                else
                    std::cout << "Checking equals: FALSE\n";

                delete[] arrayRamTmp;
                std::cout << "--------------------------------\n";
            }

            delete[] arrayRam;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;            
        }
    }



    template<typename T>
    static T ScalarProductRamRamSeq(T* arrayRam1, T* arrayRam2, size_t length)
    {
        T scalarProduct{0};

        for (size_t i = 0; i < length; i++)
        {
            scalarProduct += arrayRam1[i] * arrayRam2[i];
        }

        return scalarProduct;
    }

    template<typename T>
    static T ScalarProductRamRamParThread(T* arrayRam1, T* arrayRam2, size_t length, unsigned threadsNum)
    {
        T scalarProduct{0};
        std::mutex mutex;

        size_t blockSize = length / threadsNum;
        /*
        Пусть n - количество элементов массива, m - на сколько частей надо поделить.
        Тогда всего в m-n%m массивах будет по n/m  элементов,
        а в n%m массивах - по n/m+1 элементов.
        пример: n=105, m=10:
        в 105/10-105%10=5 по 105/10=10 элементов,
        в 105%10=5 по 105/10+1=11 элементов.
        */
        throw std::runtime_error("NOT REALIZED!");
        for (size_t i = 0; i < length; i++)
        {
            scalarProduct += arrayRam1[i] * arrayRam2[i];
        }

        return scalarProduct;
    }

    /// @brief Скалярное произведение векторов, расположенных в RAM
    static void ScalarProductRamRamSeq_ConsoleUI()
    {
        std::cout << "ScalarProductRamRamSeq_ConsoleUI\n";
        
        size_t size    = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array size: ");
        double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");

        // Инициализируем массив в RAM
        double* arrayRam1 = new double[size];
        double* arrayRam2 = new double[size];
        for (size_t i = 0; i < size; i++)
        {
            arrayRam1[i] = value;
            arrayRam2[i] = 1/value;
        }

        auto start = high_resolution_clock::now();
        double scalarProduct = ScalarProductRamRamSeq(arrayRam1, arrayRam2, size);
        auto stop = high_resolution_clock::now();                
                    
        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();                
        std::cout << "Time, mks: " << t << std::endl;
        
        std::cout << "scalarProduct: " << scalarProduct << std::endl;
                         
        delete[] arrayRam1;
        delete[] arrayRam2;
    }

    
    /// @brief Скалярное произведение векторов, расположенных в RAM, параллельно, std::thread
    static void ScalarProductRamRamParThread_ConsoleUI()
    {
        std::cout << "ScalarProductRamRamParThread_ConsoleUI\n";
        
        size_t length    = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length: ");
        //double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");
        double value   = 0.1;
        size_t threadsNum = ConsoleHelper::GetUnsignedIntFromUser("Enter number of threads: ");
        

        // Инициализируем массив в RAM
        double* arrayRam1 = new double[length];
        double* arrayRam2 = new double[length];
        for (size_t i = 0; i < length; i++)
        {
            arrayRam1[i] = value;
            arrayRam2[i] = 1/value;
        }

        auto start = high_resolution_clock::now();
        double scalarProduct = ScalarProductRamRamParThread(arrayRam1, arrayRam2, length, threadsNum);
        auto stop = high_resolution_clock::now();                
                    
        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();                
        std::cout << "Time, mks: " << t << std::endl;
        
        std::cout << "scalarProduct: " << scalarProduct << std::endl;
                         
        delete[] arrayRam1;
        delete[] arrayRam2;
    }

};

