#pragma once

#include <iostream>

/////////////// CUDA ////////////////

/// @brief Структура для хранения параметров видеокарты
struct CudaDeviceProperties
{
    bool IsInitialized = false;

    int Major;
    int Minor;
    std::string Name;
    size_t TotalGlobalMem;
    size_t SharedMemoryPerBlock;
    size_t RegsPerBlock;
    int WarpSize;
    size_t MemPitch;
    size_t MaxThreadsPerBlock;
    int MultiProcessorCount;
    bool DeviceOverlap;
    int AsyncEngineCount;// Number of asynchronous engines
    size_t MemoryClockRate;//Memory Clock Rate (KHz)
    int MemoryBusWidth;//Memory Bus Width (bits)
    
    double GetPeakMemoryBandwidthGBs()
    {
        return 2.0*MemoryClockRate*(MemoryBusWidth/8)/1.0e6;
    }
                 

    void Print()
    {
        if(!IsInitialized)
        {
            std::cout << "CudaDeviceProperties object is not initialized!" << std::endl;
            return;
        }            

        std::cout << "Major revision number:         " <<  Major                << std::endl;
        std::cout << "Minor revision number:         " <<  Minor                << std::endl;
        std::cout << "Name:                          " <<  Name                 << std::endl;
        std::cout << "Total global memory:           " <<  TotalGlobalMem       << std::endl;
        std::cout << "Total shared memory per block: " <<  SharedMemoryPerBlock << std::endl;
        std::cout << "Total registers per block:     " <<  RegsPerBlock         << std::endl;
        std::cout << "Warp size:                     " <<  WarpSize             << std::endl;
        std::cout << "Maximum memory pitch:          " <<  MemPitch             << std::endl;
        std::cout << "Maximum threads per block:     " <<  MaxThreadsPerBlock   << std::endl;
        /*for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
        for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
        printf("Clock rate:                    %d\n",  devProp.clockRate);
        printf("Total constant memory:         %u\n",  devProp.totalConstMem);
        printf("Texture alignment:             %u\n",  devProp.textureAlignment);
        printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));*/
        std::cout << "Number of multiprocessors:     " <<  MultiProcessorCount  << std::endl;
        //printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        std::cout << "Number of asynchronous engines: " <<  AsyncEngineCount           << std::endl;
        std::cout << "Memory Clock Rate (KHz):        " << MemoryClockRate             << std::endl;
        std::cout << "Memory Bus Width (bits):        " << MemoryBusWidth              << std::endl;
        std::cout << "Peak Memory Bandwidth (GB/s):   " << GetPeakMemoryBandwidthGBs() << std::endl;
    }
};

/// @brief Класс для хранения вспомогательных функций Cuda
struct CudaHelper
{
    static bool IsCudaSupported()
    {
        #ifdef __NVCC__
        return true;        
        #else
        return false;
        #endif
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
        #else
        out << "printDevProp(): CUDA is not supported!" << std::endl;
        #endif
    }


};
/////////////////// CUDA (END) /////////////////////////
