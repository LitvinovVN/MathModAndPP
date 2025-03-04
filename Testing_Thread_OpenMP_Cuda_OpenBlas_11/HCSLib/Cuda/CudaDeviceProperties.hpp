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