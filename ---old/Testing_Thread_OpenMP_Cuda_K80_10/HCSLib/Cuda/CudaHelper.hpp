#pragma once

#include <iostream>
#include "CudaDeviceProperties.hpp"
/////////////// CUDA ////////////////
#ifdef __NVCC__

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - THROWING EXCEPTION ***\n"); \
        throw std::runtime_error(msg); \
    } \
  } while (0)

#endif

/// @brief Класс для хранения вспомогательных функций Cuda
struct CudaHelper
{
    /// @brief Определяет поддержку CUDA
    /// @return true - поддерживается, false - не поддерживается
    static bool IsCudaSupported()
    {
        bool isCudaSupported = false;

        #ifdef __NVCC__
        isCudaSupported = true;
        #endif 

        return isCudaSupported;
    }

    /// @brief Возвращает количество Cuda-совместимых устройств
    /// @return Количество Cuda-совместимых устройств
    static int GetCudaDeviceNumber()
    {
        int devCount = 0;
        #ifdef __NVCC__
        cudaGetDeviceCount(&devCount);        
        #endif

        return devCount;
    }

    /// @brief Устанавливает текущее устройство (GPU)
    /// @param deviceId Индекс устройства
    /// @return true - успех
    static bool SetDevice(unsigned deviceId)
    {
        #ifdef __NVCC__
        cudaSetDevice(deviceId);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return false;
        return true;
        #else
        return false;       
        #endif
    }
    
    /// @brief Возвращает структуру с параметрами видеокарты
    /// @param deviceId Идентификатор Cuda-устройства
    /// @return Объект CudaDeviceProperties с параметрами видеокарты (поле IsInitialized = true в случае успеха, иначе IsInitialized = false)
    static CudaDeviceProperties GetCudaDeviceProperties(int deviceId = 0)
    {
        CudaDeviceProperties prop;
        #ifdef __NVCC__        
        // Get device properties
        printf("\nCUDA Device #%d\n", deviceId);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, deviceId);
        
        prop.IsInitialized = true;
        prop.Major = devProp.major;
        prop.Minor = devProp.minor;        
        prop.Name = std::string(devProp.name);        
        prop.TotalGlobalMem = devProp.totalGlobalMem;
        prop.SharedMemoryPerBlock = devProp.sharedMemPerBlock;
        prop.RegsPerBlock = devProp.regsPerBlock;
        prop.WarpSize = devProp.warpSize;
        prop.MemPitch = devProp.memPitch;
        prop.MaxThreadsPerBlock = devProp.maxThreadsPerBlock;
        //for (int i = 0; i < 3; ++i)
        //    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        //for (int i = 0; i < 3; ++i)
        //    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        //printf("Clock rate:                    %d\n",  devProp.clockRate);
        //printf("Total constant memory:         %u\n",  devProp.totalConstMem);
        //printf("Texture alignment:             %u\n",  devProp.textureAlignment);
        prop.DeviceOverlap = devProp.deviceOverlap;
        prop.MultiProcessorCount = devProp.multiProcessorCount;
        //printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));//*/
        prop.AsyncEngineCount = devProp.asyncEngineCount;
        prop.MemoryClockRate = devProp.memoryClockRate;
        prop.MemoryBusWidth = devProp.memoryBusWidth;        
        #endif
        return prop;
    }

    /// @brief Возвращает имя GPU по переданному идентификатору
    /// @param deviceId Идентификатор GPU
    /// @return Имя GPU
    static std::string GetCudaDeviceName(int deviceId = 0)
    {
        #ifdef __NVCC__
        auto cudaDeviceProperties = GetCudaDeviceProperties(deviceId);
        return cudaDeviceProperties.Name;
        #else
        std::cout << "GetCudaDeviceName(): CUDA is not supported!" << std::endl;
        return "";
        #endif
    }

    // Print device properties
    static void PrintCudaDeviceProperties(int deviceId = 0)
    {
        #ifdef __NVCC__
        // Get device properties
        std::cout << "\nCUDA Device #"                 << deviceId                      << std::endl;
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, deviceId);
        
        std::cout << "Major revision number:          " << devProp.major                << std::endl;
        std::cout << "Minor revision number:          " << devProp.minor                << std::endl;
        std::cout << "Name:                           " << devProp.name                 << std::endl;
        std::cout << "Total global memory:            " << devProp.totalGlobalMem       << std::endl;
        std::cout << "Total shared memory per block:  " << devProp.sharedMemPerBlock    << std::endl;
        std::cout << "Total registers per block:      " << devProp.regsPerBlock         << std::endl;
        std::cout << "Warp size:                      " << devProp.warpSize             << std::endl;
        std::cout << "Maximum memory pitch:           " << devProp.memPitch             << std::endl;
        std::cout << "Maximum threads per block:      " << devProp.maxThreadsPerBlock   << std::endl;
        /*for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %u\n",  devProp.totalConstMem);
        printf("Texture alignment:             %u\n",  devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));*/
        std::cout << "Number of multiprocessors:      " << devProp.multiProcessorCount  << std::endl;
        //printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        std::cout << "Number of asynchronous engines: " <<  devProp.asyncEngineCount         << std::endl;
        std::cout << "Memory Clock Rate (KHz):        " << devProp.memoryClockRate           << std::endl;
        std::cout << "Memory Bus Width (bits):        " << devProp.memoryBusWidth            << std::endl;
        std::cout << "Peak Memory Bandwidth (GB/s):   " << 2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6 << std::endl;
        #else
        std::cout << "printDevProp(): CUDA is not supported!" << std::endl;
        #endif
    }

    static void PrintCudaDeviceProperties_ConsoleUI()
    {
        int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();
        std::cout << "CudaDeviceNumber: " 
                  << cudaDeviceNumber 
                  << std::endl;
        std::cout << "Enter deviceId (0..." << cudaDeviceNumber-1 << "): ";
        int deviceId;
        std::cin >> deviceId;
        PrintCudaDeviceProperties(deviceId);
    }

    static void WriteGpuSpecs(std::ofstream& out)
    {
        #ifdef __NVCC__
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
        //#ifdef __NVCC__
        #else
        out << "printDevProp(): CUDA is not supported!" << std::endl;
        #endif
    }

    /// @brief Записывает параметры видеокарты в текстовый файл gpu-specs.txt
    static void WriteGpuSpecsToTxtFile_ConsoleUI()
    {
        int cudaDeviceNumber = CudaHelper::GetCudaDeviceNumber();
        std::cout << "Cuda devices number: " << cudaDeviceNumber << std::endl;
        //CudaHelper::PrintCudaDeviceProperties();

        if(cudaDeviceNumber > 0)
        {
            for(int i = 0; i < cudaDeviceNumber; i++)
            {
                auto devProps = CudaHelper::GetCudaDeviceProperties();
                devProps.Print();
            }
            
            std::ofstream f("gpu-specs.txt");
            CudaHelper::WriteGpuSpecs(f);
            f.close();
        }
    }

    template<typename T>
    static void CudaFree(T* dev_arr)
    {
        #ifdef __NVCC__
        cudaFree(dev_arr);
        #endif
    }

    template<typename T>
    static void InitByValue(T* arrayGpu,
        unsigned long long length,
        T value)
    {
        #ifdef __NVCC__
        unsigned threadsNum = 1024;
        unsigned blocksNum  = length / threadsNum;
        if(length % threadsNum)
            blocksNum++;
        kernel_init_by_value<<<blocksNum, threadsNum>>>(arrayGpu, length, value);
        cudaError_t cudaResult = cudaGetLastError();
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not init GPU array by value: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }
        #else
        std::cout << "CUDA is not supported!" << std::endl;
        #endif
    }

    template<typename T, typename S>
    static void Multiply(T* arrayGpu, unsigned long long length, S scalar)
    {
        #ifdef __NVCC__
        unsigned threadsNum = 1024;
        unsigned blocksNum  = length / threadsNum;
        if(length % threadsNum)
            blocksNum++;
        kernel_multiply<<<blocksNum, threadsNum>>>(arrayGpu, length, scalar);
        cudaDeviceSynchronize();
        #else
        std::cout << "CUDA is not supported!" << std::endl;
        #endif
    }
};
/////////////////// CUDA (END) /////////////////////////
