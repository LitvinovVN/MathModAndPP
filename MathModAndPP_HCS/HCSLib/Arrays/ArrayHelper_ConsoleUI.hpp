#pragma once

#include <iostream>
#include <chrono>

#include "../CommonHelpers/ConsoleHelper.hpp"
#include "../Cuda/CudaHelper.hpp"
#include "ArrayHelper.hpp"

/// @brief Структура для хранения консольного пользовательского интерфейса для методов класса ArrayHelper, обрабатывающих массивы T*.
struct ArrayHelper_ConsoleUI
{
    /// @brief Выделение закрепленной памяти
    static void CreateArrayRamPinned_ConsoleUI()
    {
        std::cout << "CreateArrayRamPinned_ConsoleUI\n";
        if(!CudaHelper::IsCudaSupported())
        {
            std::cout << "Cuda not supported!" << std::endl;
            return;
        }

        size_t length  = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length: ");
        double* arrPinned = ArrayHelper::CreateArrayRamPinned<double>(length);
        if(!arrPinned)
        {
            std::cout << "Pinned RAM memory not allocated!" << std::endl;
            return;
        }

        arrPinned[0] = 0.1;
        std::cout << "arrPinned[0] = 0.1;\n";
        std::cout << "arrPinned[0] = " << arrPinned[0] <<";\n";

        ArrayHelper::DeleteArrayRamPinned(arrPinned);
        std::cout << "Pinned RAM memory cleared!\n";
    }

    /// @brief Копирование данных из RAM в GPU
    static void CopyRamToGpu_ConsoleUI()
    {
        std::cout << "CopyRamToGpu_ConsoleUI\n";
        try
        {
            int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();            
            std::cout << "cudaDeviceNumber: " << cudaDeviceNumber << std::endl;
            
            size_t length  = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length: ");
            //double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");
            double value   = 0;

            int memoryType = ConsoleHelper::GetIntFromUser("Enter type of ram alloc (1-paged; 2-pinned): ");

            // Инициализируем массив в RAM
            double* arrayRam = nullptr;
            if(memoryType == 1)
                arrayRam = new double[length];
            else if (memoryType == 2)
                arrayRam = ArrayHelper::CreateArrayRamPinned<double>(length);
            else
            {
                std::cout << "type of ram alloc not recognized: " << memoryType << std::endl;
                return;
            }

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
                double* arrayGpu = ArrayHelper::CreateArrayGpu<double>(length, i);
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
                    ArrayHelper::PrintArrayRam(arrayRam, 0, 20);
                else
                    ArrayHelper::PrintArrayRam(arrayRam, 0, length);

                std::cout << "GPU " << i << ": ";                
                if(length>20)                
                    ArrayHelper::PrintArrayGpu(arrayGpu, 0, 20, i);
                else
                    ArrayHelper::PrintArrayGpu(arrayGpu, 0, length, i);

                //arrayRam[length-1]+=0.00001;
                bool isEquals = ArrayHelper::IsEqualsRamGpu(arrayRam, arrayGpu, length);
                if (isEquals)
                    std::cout << "Success! Arrays are equals!\n";
                else
                    std::cout << "Error! Arrays are not equals!\n";
                
                std::cout << "--------------------------------\n";
                ArrayHelper::DeleteArrayGpu(arrayGpu, i);
            }

            
            if(memoryType == 1)
                delete[] arrayRam;
            else if (memoryType == 2)
                ArrayHelper::DeleteArrayRamPinned(arrayRam);
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
                double* dev_array = ArrayHelper::CreateArrayGpu<double>(size, i);
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
                    ArrayHelper::PrintArrayRam(arrayRam, 0, 20);
                else
                    ArrayHelper::PrintArrayRam(arrayRam, 0, size);

                std::cout << "GPU " << i << ": ";                
                if(size>20)                
                    ArrayHelper::PrintArrayGpu(dev_array, 0, 20, i);
                else
                    ArrayHelper::PrintArrayGpu(dev_array, 0, size, i);
                
                std::cout << "Ram copied: ";
                if(size>20)
                    ArrayHelper::PrintArrayRam(arrayRamTmp, 0, 20);
                else
                    ArrayHelper::PrintArrayRam(arrayRamTmp, 0, size);
                
                bool isEquals = ArrayHelper::IsEqualsRamRam(arrayRam, arrayRamTmp, size);
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

    /// @brief Работа с функцией SumOpenMP
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

    /// @brief Работа с функцией SumCudaMultiGpu(std::vector<ArrayGpuProcessingParams<T>> params)
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
                    param.dev_arr = ArrayHelper::CreateArrayGpu<double>(size, i);
                    std::cout << "array " << i << " created\n";
                    std::cout << "First 10 elements of " << i << " array: ";                
                    ArrayHelper::PrintArrayGpu(param.dev_arr, 0, 10, i);

                    ArrayHelper::InitArrayGpu(param.dev_arr, size, value, i);
                    std::cout << "array " << i << " initialized\n";
                    std::cout << "First 10 elements of " << i << " array: ";                
                    ArrayHelper::PrintArrayGpu(param.dev_arr, 0, 10, i);
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

    /// @brief Работа с функцией SumCublas
    static void SumCublas_ConsoleUI()
    {
        std::cout << "SumCublas(...)\n";
        // Вызов функции суммирования с помощью Cuda на нескольких GPU
        try
        {
            bool isCudaSupported = CudaHelper::IsCudaSupported();
            if(!isCudaSupported)
            {
                std::cout << "Cuda is not supported!" << std::endl;
                return;
            }

            int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();
            //cudaDeviceNumber = 1;
            int deviceId = 0;
            std::cout << "cudaDeviceNumber: " << cudaDeviceNumber << std::endl;
            size_t size  = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array size: ");
            //size_t size   = 200000000ull;
            std::cout << "size: " << size << std::endl;
            //double value = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");
            double value   = 0.001;
            //int blocksNum  = ConsoleHelper::GetIntFromUser("Enter num blocks: ");
            //int threadsNum = ConsoleHelper::GetIntFromUser("Enter num threads: ");            
            int blocksNum  = 34;
            int threadsNum = 16;

            ArrayGpuProcessingParams<double> params;
            params.deviceId   = deviceId;
            params.indStart   = 0;
            params.indEnd     = size-1;
            params.blocksNum  = blocksNum;
            params.threadsNum = threadsNum;
            params.dev_arr = ArrayHelper::CreateArrayGpu<double>(size, params.deviceId);
            std::cout << "Array on device " << params.deviceId << " created!\n";
            std::cout << "First 10 elements: ";                
            ArrayHelper::PrintArrayGpu(params.dev_arr, 0, 10, params.deviceId);

            ArrayHelper::InitArrayGpu(params.dev_arr, size, value, params.deviceId);
            std::cout << "array " << params.deviceId << " initialized\n";
            std::cout << "First 10 elements of " << params.deviceId << " array: ";                
            ArrayHelper::PrintArrayGpu(params.dev_arr, 0, 10, params.deviceId);
                                
            std::cout << "Initializing array completed!\n";

            cublasHandle_t cublasH = CublasHelper::CublasCreate();

            double sum = 0;
            auto start = high_resolution_clock::now();
            sum = ArrayHelper::SumCublas(cublasH, params);
            auto stop = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(stop - start);
            auto t = duration.count();

            std::cout << "ArrayHelper::SumCuBLAS(...): " << sum << std::endl;
            std::cout << "Expected sum: " << size*value << std::endl;
            std::cout << "Time, mks: " << t << std::endl;
            CudaHelper::CudaFree(params.dev_arr);
            CublasHelper::CublasDestroy(cublasH);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    /// @brief Работа с функцией SumCublasMultiGpu(std::vector<ArrayGpuProcessingParams<T>> params)
    static void SumCublasMultiGpu_ConsoleUI()
    {
        std::cout << "SumCublasMultiGpu(std::vector<ArrayGpuProcessingParams<T>> params)\n";
        // Вызов функции суммирования с помощью Cublas на нескольких GPU
        try
        {
            int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();
            //cudaDeviceNumber = 1;
            std::cout << "cudaDeviceNumber: " << cudaDeviceNumber << std::endl;
            double expectedResult = 0;
            
            std::vector<cublasHandle_t> cublasHandles;
            std::vector<double*> dev_arrays;
            std::vector<size_t> indStarts;
            std::vector<size_t> indEnds;

            for(int i = 0; i < cudaDeviceNumber; i++)
            {
                cublasHandle_t cublasHandle = CublasHelper::CublasCreate(i);
                cublasHandles.push_back(cublasHandle);

                std::cout << "--- Init " << i << " array starting...\n";
                //size_t size    = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array size: ");
                //double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");
                size_t size    = 500000000ull;
                double value   = 0.001;                
                
                expectedResult += size*value;
                
                try
                {
                    double* dev_arr = ArrayHelper::CreateArrayGpu<double>(size, i);
                    std::cout << "array " << i << " created\n";
                    std::cout << "First 10 from " << size <<" elements of " << i << " array: ";                
                    ArrayHelper::PrintArrayGpu(dev_arr, 0, 10, i);

                    ArrayHelper::InitArrayGpu(dev_arr, size, value, i);
                    std::cout << "array " << i << " initialized\n";
                    std::cout << "First 10 from " << size <<" elements of " << i << " array: ";                
                    ArrayHelper::PrintArrayGpu(dev_arr, 0, 10, i);

                    std::cout << "--- Initializing " << i << " array completed!\n";

                    dev_arrays.push_back(dev_arr);
                    indStarts.push_back(0);
                    indEnds.push_back(size-1);
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                    std::exit(-1);
                }
            }
            
            auto start = high_resolution_clock::now();
            double sum = ArrayHelper::SumCublasMultiGpu(cublasHandles,
                dev_arrays, indStarts, indEnds);
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


    /// @brief Скалярное произведение векторов, расположенных в RAM
    static void ScalarProductRamSeq_ConsoleUI()
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
        double scalarProduct = ArrayHelper::ScalarProductRamSeq(arrayRam1, arrayRam2, size);
        auto stop = high_resolution_clock::now();                
                    
        auto duration = duration_cast<microseconds>(stop - start);        
        auto t = duration.count();                
        std::cout << "Time, mks: " << t << std::endl;
        
        std::cout << "scalarProduct: " << scalarProduct << std::endl;
                         
        delete[] arrayRam1;
        delete[] arrayRam2;
    }

    /// @brief Скалярное произведение векторов, расположенных в RAM, параллельно, std::thread
    static void ScalarProductRamParThread_ConsoleUI()
    {
        try
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
            double scalarProduct = ArrayHelper::ScalarProductRamParThread(arrayRam1, arrayRam2, length, threadsNum);
            auto stop = high_resolution_clock::now();                
                        
            auto duration = duration_cast<microseconds>(stop - start);        
            auto t = duration.count();                
            std::cout << "Time, mks: " << t << std::endl;
            
            std::cout << "scalarProduct: " << scalarProduct << std::endl;
                            
            delete[] arrayRam1;
            delete[] arrayRam2;
        }
        catch(const std::exception& e)
        {
            std::cout << e.what() << '\n';
        }
    }

    
    /// @brief Скалярное произведение векторов, расположенных в RAM, параллельно, OpenMP
    static void ScalarProductRamParOpenMP_ConsoleUI()
    {
        try
        {
            std::cout << "ScalarProductRamParOpenMP_ConsoleUI\n";
            
            size_t length    = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length: ");
            //double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");
            double value   = 0.1;
            size_t threadsNum = ConsoleHelper::GetUnsignedIntFromUser("Enter number of OpenMP threads: ");
            

            // Инициализируем массив в RAM
            double* arrayRam1 = new double[length];
            double* arrayRam2 = new double[length];
            for (size_t i = 0; i < length; i++)
            {
                arrayRam1[i] = value;
                arrayRam2[i] = 1/value;
            }

            auto start = high_resolution_clock::now();
            double scalarProduct = ArrayHelper::ScalarProductRamParOpenMP(arrayRam1, arrayRam2, length, threadsNum);
            auto stop = high_resolution_clock::now();                
                        
            auto duration = duration_cast<microseconds>(stop - start);        
            auto t = duration.count();                
            std::cout << "Time, mks: " << t << std::endl;
            
            std::cout << "scalarProduct: " << scalarProduct << std::endl;
                            
            delete[] arrayRam1;
            delete[] arrayRam2;
        }
        catch(const std::exception& e)
        {
            std::cout << e.what() << '\n';
        }
    }


    /// @brief Скалярное произведение векторов, расположенных в RAM, параллельно, OpenBlas
    static void ScalarProductRamOpenBlas_ConsoleUI()
    {
        try
        {
            std::cout << "ScalarProductRamOpenBlas_ConsoleUI\n";
            
            size_t length    = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length: ");
            //double value   = ConsoleHelper::GetDoubleFromUser("Enter value: ","Error! Enter double value");
            double value   = 0.1;
            //size_t threadsNum = ConsoleHelper::GetUnsignedIntFromUser("Enter number of OpenMP threads: ");
            

            // Инициализируем массив в RAM
            double* arrayRam1 = new double[length];
            double* arrayRam2 = new double[length];
            for (size_t i = 0; i < length; i++)
            {
                arrayRam1[i] = value;
                arrayRam2[i] = 1/value;
            }

            std::cout << "\ncblas_ddot\n";
            double scalarProduct = 0;
            
            try
            {
                auto start = high_resolution_clock::now();
                
                scalarProduct = ArrayHelper::ScalarProductRamCublas(arrayRam1, arrayRam2, length);
            
                auto stop = high_resolution_clock::now();                
                        
                auto duration = duration_cast<microseconds>(stop - start);        
                auto t = duration.count();                
                std::cout << "Time, mks: " << t << std::endl;
                
                std::cout << "scalarProduct: " << scalarProduct << std::endl;
            }
            catch (const std::exception& exc)
            {
                std::cout << exc.what() << std::endl;
            }
        
            delete[] arrayRam1;
            delete[] arrayRam2;
        }
        catch(const std::exception& e)
        {
            std::cout << e.what() << '\n';
        }
    }



    /// @brief Скалярное произведение векторов, расположенных в GPU, параллельно, Cuda
    static void ScalarProductGpuParCuda_ConsoleUI()
    {
        std::cout << "ScalarProductGpuParCuda_ConsoleUI()\n";

        if(!CudaHelper::IsCudaSupported())
        {
            std::cout << "CUDA not supported!\n";
            return;
        }

        try
        {
            size_t length = ConsoleHelper::GetUnsignedLongLongFromUser("Enter arrays length: ");
            unsigned kernelBlocks  = ConsoleHelper::GetUnsignedIntFromUser("Enter number of CUDA blocks: ");
            unsigned kernelThreads = ConsoleHelper::GetUnsignedIntFromUser("Enter number of CUDA threads in block: ");
            
            auto resFloat  = ArrayHelper::ScalarProductGpuParCuda<float>(length, kernelBlocks, kernelThreads);
            std::cout << "float: ";
            resFloat.Print();

            auto resDouble = ArrayHelper::ScalarProductGpuParCuda<double>(length, kernelBlocks, kernelThreads);
            std::cout << "double: ";
            resDouble.Print();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
    }

    /// @brief Скалярное произведение векторов, расположенных в нескольких GPU, параллельно, Cuda
    static void ScalarProductMultiGpuParCuda_ConsoleUI()
    {
        std::cout << "ScalarProductMultiGpuParCuda_ConsoleUI()\n";

        if(!CudaHelper::IsCudaSupported())
        {
            std::cout << "CUDA not supported!\n";
            return;
        }

        try
        {
            size_t length = ConsoleHelper::GetUnsignedLongLongFromUser("Enter arrays length: ");
            unsigned kernelBlocks  = ConsoleHelper::GetUnsignedIntFromUser("Enter number of CUDA blocks: ");
            unsigned kernelThreads = ConsoleHelper::GetUnsignedIntFromUser("Enter number of CUDA threads in block: ");
            int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();
            
            std::vector<double> kGpuData;// Коэффициент распределения данных между GPU
            double kGpuDistrubution = 1.0;
            for (int i = 0; i < cudaDeviceNumber; i++)
            {
                std::string msg = "Enter k GPU " + std::to_string(i);
                msg += " [";
                msg += CudaHelper::GetCudaDeviceName(i);
                msg += "]";
                msg += "(0.." + std::to_string(kGpuDistrubution) + "): ";
                double kGpu = ConsoleHelper::GetDoubleFromUser(msg);
                if(kGpu<0)
                    kGpu = 0;
                else if(kGpu>kGpuDistrubution)
                    kGpu=kGpuDistrubution;
                kGpuDistrubution -= kGpu;
                kGpuData.push_back(kGpu);
                std::cout << "Accepted: " << kGpu << "; ";
                std::cout << "Remain: " << kGpuDistrubution << "\n";
            }            

            auto resFloat  = ArrayHelper::ScalarProductMultiGpuParCuda<float>(length, kernelBlocks, kernelThreads, kGpuData);
            std::cout << "float: ";
            resFloat.Print();

            auto resDouble = ArrayHelper::ScalarProductMultiGpuParCuda<double>(length, kernelBlocks, kernelThreads, kGpuData);
            std::cout << "double: ";
            resDouble.Print();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
    }

    /// @brief Скалярное произведение векторов, расположенных в GPU, параллельно, Cublas
    static void ScalarProductGpuCublas_ConsoleUI()
    {
        std::cout << "ScalarProductGpuCublas_ConsoleUI()\n";

        if(!CudaHelper::IsCudaSupported())
        {
            std::cout << "CUDA not supported!\n";
            return;
        }

        try
        {
            size_t length = ConsoleHelper::GetUnsignedLongLongFromUser("Enter arrays length: ");
            
            auto resFloat  = ArrayHelper::ScalarProductGpuCublas<float>(length);
            std::cout << "float: ";
            resFloat.Print();

            auto resDouble = ArrayHelper::ScalarProductGpuCublas<double>(length);
            std::cout << "double: ";
            resDouble.Print();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
    }

    /// @brief Работа с функцией ScalarProductMultiGpuCublas
    static void ScalarProductMultiGpuCublas_ConsoleUI()
    {
        std::cout << "ScalarProductMultiGpuCublas_ConsoleUI()\n";
        
        try
        {
            int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();
            //cudaDeviceNumber = 1;
            std::cout << "cudaDeviceNumber: " << cudaDeviceNumber << std::endl;

            if(cudaDeviceNumber < 2)
            {
                std::cout << "GPU number must be greater 2!\n";
                return;
            }
                        
            std::vector<cublasHandle_t> cublasHandles;
            std::vector<double*> dev_arrays_1;
            std::vector<double*> dev_arrays_2;
            std::vector<size_t> dev_arrays_lengths;

            size_t length = ConsoleHelper::GetUnsignedLongLongFromUser("Enter arrays length: ");
            double expectedResult = length;
            double value_1 = 0.001;
            double value_2 = 1/value_1;

            for(int i = 0; i < cudaDeviceNumber; i++)
            {
                cublasHandle_t cublasHandle = CublasHelper::CublasCreate(i);
                cublasHandles.push_back(cublasHandle);

                std::cout << "--- Init data on GPU " << i << " ---\n";
                double kGpu    = ConsoleHelper::GetDoubleFromUser("Enter kGpu (0...1): ","Error! Enter double value");
                size_t size    = length * kGpu;
                if(i==cudaDeviceNumber-1)
                    size = length - length * kGpu * i;                                              
                
                try
                {
                    double* dev_arr_1 = ArrayHelper::CreateArrayGpu<double>(size, i);
                    std::cout << "array 1 on GPU " << i << " created\n";
                    std::cout << "First 10 from " << size <<" elements of array 1 on GPU " << i << ": ";                
                    ArrayHelper::PrintArrayGpu(dev_arr_1, 0, 10, i);

                    ArrayHelper::InitArrayGpu(dev_arr_1, size, value_1, i);
                    std::cout << "array 1 on GPU " << i << " initialized\n";
                    std::cout << "First 10 from " << size <<" elements of array 2 on GPU " << i << ": ";                
                    ArrayHelper::PrintArrayGpu(dev_arr_1, 0, 10, i);

                    std::cout << "--- Initializing array 1 on GPU " << i << " completed!\n";

                    dev_arrays_1.push_back(dev_arr_1);


                    double* dev_arr_2 = ArrayHelper::CreateArrayGpu<double>(size, i);
                    std::cout << "array 2 on GPU " << i << " created\n";
                    std::cout << "First 10 from " << size <<" elements of array 2 on GPU " << i << ": ";                
                    ArrayHelper::PrintArrayGpu(dev_arr_2, 0, 10, i);

                    ArrayHelper::InitArrayGpu(dev_arr_2, size, value_2, i);
                    std::cout << "array 1 on GPU " << i << " initialized\n";
                    std::cout << "First 10 from " << size <<" elements of array 2 on GPU " << i << ": ";                
                    ArrayHelper::PrintArrayGpu(dev_arr_2, 0, 10, i);

                    std::cout << "--- Initializing array 2 on GPU " << i << " completed!\n";

                    dev_arrays_2.push_back(dev_arr_2);

                    
                    dev_arrays_lengths.push_back(size);
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                    std::exit(-1);
                }
            }
            
            auto start = high_resolution_clock::now();
            double scalarProduct = ArrayHelper::ScalarProductMultiGpuCublas(cublasHandles,
                dev_arrays_1, dev_arrays_2, dev_arrays_lengths);
            auto stop = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(stop - start);        
            auto t = duration.count();

            std::cout << "ArrayRamHelper::ScalarProductMultiGpuCublas(...): " << scalarProduct << std::endl;
            std::cout << "Expected scalarProduct: " << expectedResult << std::endl;
            std::cout << "Time, mks: " << t << std::endl;

            // Освобождение ресурсов
            for (size_t i = 0; i < dev_arrays_1.size(); i++)
            {
                ArrayHelper::DeleteArrayGpu(dev_arrays_1[i], i);
                ArrayHelper::DeleteArrayGpu(dev_arrays_2[i], i);
            }
            
            CublasHelper::CublasDestroy(cublasHandles);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;            
        }
    }


};

