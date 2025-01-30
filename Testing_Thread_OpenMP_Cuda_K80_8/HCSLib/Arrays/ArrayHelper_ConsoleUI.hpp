#pragma once

#include <iostream>
#include <chrono>

#include "../CommonHelpers/ConsoleHelper.hpp"
#include "../Cuda/CudaHelper.hpp"
#include "ArrayHelper.hpp"

/// @brief Структура для хранения консольного пользовательского интерфейса для методов класса ArrayHelper, обрабатывающих массивы T*.
struct ArrayHelper_ConsoleUI
{

    /// @brief Копирование данных из RAM в GPU
    static void CopyRamToGpu_ConsoleUI()
    {
        std::cout << "CopyRamToGpu_ConsoleUI\n";
        try
        {
            int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();            
            std::cout << "cudaDeviceNumber: " << cudaDeviceNumber << std::endl;
            
            size_t length  = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length: ");
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
            //size_t size  = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array size: ");
            size_t size   = 200000000ull;
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


};

