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


    ////////////////////////// Создание и освобождение массивов (начало) /////////////////////////////

    /// @brief Выделяет память для массива в RAM
    /// @tparam T Тип элементов массива
    /// @param size Количество элементов
    /// @return Указатель на созданный массив
    template<typename T>
    static T* CreateArrayRam(unsigned long long size)
    {
        return new T[size];
    }

    template<typename T>
    static void DeleteArrayRam(T* arrayRam)
    {
        if(arrayRam == nullptr)
            return;

        delete[] arrayRam;
        arrayRam = nullptr;
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
        
        //std::cout << "Allocating GPU memory: " << size << " * " << sizeof(T) << " = " << size*sizeof(T) << " bytes... ";
        T* dev_array = nullptr;
        cudaMalloc(&dev_array, size*sizeof(T));
        
        std::string msg("Could not allocate device memory for GPU array: ");
        msg += std::to_string(size*sizeof(T));
        msg += " bytes not allocated!\n";
        cudaCheckErrors(msg.c_str());        
        cudaDeviceSynchronize();

        std::cout << "OK\n";
        
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


    /// @brief Освобождает массив на текущем GPU
    /// @tparam T Указатель на массив в GPU
    template<typename T>
    static void DeleteArrayGpu(T* arrayGpu)
    {
        #ifdef __NVCC__
        //std::cout << "Clearing gpu array " << arrayGpu << ": ";
        if (arrayGpu == nullptr)
            return;

        cudaFree(arrayGpu);
        cudaCheckErrors("Error in cudaFree!");
        arrayGpu = nullptr;
        //std::cout << "OK (" << arrayGpu << ")\n";
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    /// @brief Освобождает массив на GPU c
    /// @tparam T Указатель на массив в GPU
    /// @tparam deviceId Идентификатор GPU
    template<typename T>
    static void DeleteArrayGpu(T* arrayGpu, int deviceId)
    {
        #ifdef __NVCC__
        if (deviceId == 0)
        {
            DeleteArrayGpu(arrayGpu);
            return;
        }

        std::thread th{
            [&](){
                cudaSetDevice(deviceId);
                cudaCheckErrors("Error in cudaSetDevice!");
                DeleteArrayGpu(arrayGpu);
            }
        };

        th.join();
        
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }
    ////////////////////////// Создание и освобождение массивов (конец) /////////////////////////////


    ////////////////////////// Инициализация массивов (начало) /////////////////////////////

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
                //std::cout << "Init block " << blockIndex << std::endl;

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
                //std::cout << "Init last block " << lastBlockSize << " elements" << std::endl;

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

    ////////////////////////// Инициализация массивов (конец) /////////////////////////////


    ////////////////////////// Копирование массивов (начало) /////////////////////////////
    
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

    ////////////////////////// Копирование массивов (конец) /////////////////////////////


    ////////////////////////// Сравнение массивов (начало) /////////////////////////////

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
    
    ////////////////////////// Сравнение массивов (конец) /////////////////////////////




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

    // Суммирование на нескольких GPU
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
                       
        //#ifdef __NVCC__
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    ////////////////////////// Суммирование элементов массива (конец) /////////////////////////////

    
    ////////////////////////// Скалярное произведение элементов массива (начало) /////////////////////////////
    
    template<typename T>
    static T ScalarProductRamSeq(T* arrayRam1, T* arrayRam2, size_t length)
    {
        T scalarProduct{0};

        for (size_t i = 0; i < length; i++)
        {
            scalarProduct += arrayRam1[i] * arrayRam2[i];            
        }
        return scalarProduct;
    }

    template<typename T>
    static T ScalarProductRamParThread(T* arrayRam1, T* arrayRam2, size_t length, unsigned threadsNum)
    {
        T scalarProduct{0};
        std::mutex mutex;
        std::vector<std::thread> threads;

        size_t blockSize = length / threadsNum;
            
        for (size_t i = 0; i < threadsNum; i++)
        {
            threads.push_back(
                std::thread{
                    [=, &mutex, &scalarProduct](){
                        T localSum = ScalarProductRamSeq(arrayRam1 + i*blockSize,
                            arrayRam2 + i*blockSize,
                            (i < threadsNum-1) ? blockSize : blockSize + length % threadsNum);
                        
                        {
                            std::lock_guard<std::mutex> guard{mutex};
                            scalarProduct += localSum;
                        }
                    }
                }
            );
            
        }

        for(auto& thread : threads)
        {
            thread.join();
        }

        return scalarProduct;
    }

    template<typename T>
    static T ScalarProductGpuParCuda(T* arrayGpu1, T* arrayGpu2, size_t length,
        unsigned kernelBlocks, unsigned kernelThreads)
    {
        #ifdef __NVCC__

        // Выделяем в распределяемой памяти каждого SM массив для хранения локальных сумм каждого потока блока
        unsigned shared_mem_size = kernelThreads * sizeof(T);
        // Выделяем в RAM и глобальной памяти GPU массив для локальных сумм каждого блока
        T* blockSumsRam = CreateArrayRam<T>(kernelBlocks);
        T* blockSumsGpu = CreateArrayGpu<T>(kernelBlocks);
        // Запуск ядра вычисления скалярного произведения
        kernel_scalar_product<<<kernelBlocks, kernelThreads, shared_mem_size>>>(arrayGpu1, arrayGpu2, length, blockSumsGpu);
        cudaCheckErrors("Error in kernel_scalar_product!\n");
        // Копируем частичные суммы из GPU в RAM
        std::cout << "Starting CopyGpuToRam... ";
        CopyGpuToRam(blockSumsGpu, blockSumsRam, kernelBlocks);
        std::cout << "OK\n";
        T result = Sum(blockSumsRam, kernelBlocks);
        // Освобождаем память        
        DeleteArrayRam(blockSumsRam);
        DeleteArrayGpu(blockSumsGpu);
        return result;

        //#ifdef __NVCC__
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    template<typename T>
    static FuncResult<T> ScalarProductGpuParCuda(size_t length,
        unsigned kernelBlocks, unsigned kernelThreads)
    {
        #ifdef __NVCC__

        T* arrayGpu1 = CreateArrayGpu<T>(length);
        T* arrayGpu2 = CreateArrayGpu<T>(length);
        ArrayHelper::InitArrayGpu(arrayGpu1, length, (T)10.0);
        ArrayHelper::InitArrayGpu(arrayGpu2, length, (T)0.1);

        auto start = high_resolution_clock::now();
        T scalarProduct = ScalarProductGpuParCuda(arrayGpu1, arrayGpu2, length, kernelBlocks, kernelThreads);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);
        auto t = duration.count();
        
        DeleteArrayGpu(arrayGpu1);
        DeleteArrayGpu(arrayGpu2);

        return FuncResult<T>{true, scalarProduct, t};

        //#ifdef __NVCC__
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }



    template<typename T>
    static T ScalarProductMultiGpuParCuda(
        std::vector<T*> array1Gpus,
        std::vector<T*> array2Gpus,
        std::vector<size_t> lengthGpus,
        unsigned kernelBlocks,
        unsigned kernelThreads)
    {
        #ifdef __NVCC__
        
        T scalarProduct{0};
        
        auto gpuNum = array1Gpus.size();

        std::vector<std::thread> threads;
        std::mutex mutex;
        for(int i = 0; i < gpuNum; i++)
        {
            threads.push_back(std::thread{
                [i, &mutex, &array1Gpus, &array2Gpus,
                    &lengthGpus, kernelBlocks, kernelThreads, &scalarProduct](){
                cudaSetDevice(i);
                
                std::cout   << i << "; "
                            << array1Gpus[i] << "; "
                            << array2Gpus[i] << "; "
                            << lengthGpus[i] << "; "
                            << kernelBlocks << "; "
                            << kernelThreads
                            << std::endl;

                T gpu_scalarProduct = ScalarProductGpuParCuda(
                    array1Gpus[i],
                    array2Gpus[i],
                    lengthGpus[i],
                    kernelBlocks,
                    kernelThreads
                );
                mutex.lock();
                //std::cout << "thread " << i <<": ";
                //params[i].Print();
                //std::cout << "scalarProduct = " << scalarProduct <<"\n";
                scalarProduct += gpu_scalarProduct;
                mutex.unlock();
            }});
        }       

        for(auto& thread : threads)
        {
            thread.join();
        }

        return scalarProduct;
                       
        //#ifdef __NVCC__
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }


    
    template<typename T>
    static FuncResult<T> ScalarProductMultiGpuParCuda(size_t length,
        unsigned kernelBlocks, unsigned kernelThreads, std::vector<double> kGpuData)
    {
        #ifdef __NVCC__

        std::vector<size_t> gpuDataLength;
        size_t gpuDataLengthDistribution = length;
        int gpuNum = kGpuData.size();
        for (size_t i = 0; i < gpuNum; i++)
        {
            std::cout << "kGpuData[" << i << "]: " << kGpuData[i] << "\n";

            size_t gpuDataLengthElement = kGpuData[i] * length;
            if(i == gpuNum - 1)
            {
                gpuDataLengthElement = gpuDataLengthDistribution;
            }
            gpuDataLength.push_back(gpuDataLengthElement);
            gpuDataLengthDistribution -= gpuDataLengthElement;
        }
                
        std::vector<T*> array1Gpus;
        std::vector<T*> array2Gpus;

        for (size_t i = 0; i < gpuDataLength.size(); i++)
        {
            std::cout << "GPU " << i << ": "
                      << gpuDataLength[i]
                      << " from " << length << std::endl;
            T* array1Gpu = CreateArrayGpu<T>(gpuDataLength[i], i);
            array1Gpus.push_back(array1Gpu);
            T* array2Gpu = CreateArrayGpu<T>(gpuDataLength[i], i);
            array2Gpus.push_back(array2Gpu);
        }
        std::cout << "Arrays created!\n";


        for (size_t i = 0; i < gpuDataLength.size(); i++)
        {
            std::cout << "GPU " << i << ": "
                      << gpuDataLength[i]
                      << " from " << length << std::endl;
            ArrayHelper::InitArrayGpu(array1Gpus[i], gpuDataLength[i], (T)10.0, i);
            ArrayHelper::InitArrayGpu(array2Gpus[i], gpuDataLength[i], (T)0.1, i);
        }
        std::cout << "Arrays initialized!\n";

        

        auto start = high_resolution_clock::now();
        T scalarProduct = ScalarProductMultiGpuParCuda(array1Gpus, array2Gpus, gpuDataLength, kernelBlocks, kernelThreads);
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);
        auto t = duration.count();
        
        
        for (size_t i = 0; i < gpuDataLength.size(); i++)
        {
            std::cout << "GPU " << i << ": "
                      << gpuDataLength[i]
                      << " from " << length << std::endl;
            DeleteArrayGpu(array1Gpus[i], i);
            DeleteArrayGpu(array2Gpus[i], i);
        }
        std::cout << "Arrays deleted!\n";

        return FuncResult<T>{true, scalarProduct, t};

        //#ifdef __NVCC__
        #else
            throw std::runtime_error("CUDA not supported!");
        #endif
    }

    ////////////////////////// Скалярное произведение элементов массива (конец) /////////////////////////////


    /*  ---   Другие алгоритмы   ---  */





};

