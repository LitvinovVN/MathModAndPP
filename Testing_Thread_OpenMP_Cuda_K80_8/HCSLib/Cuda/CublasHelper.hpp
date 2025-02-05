#pragma once

#ifndef __NVCC__
struct cublasHandle_t{};
struct cublasStatus_t{};
#endif

/// @brief Класс для хранения вспомогательных функций CuBLAS
struct CublasHelper
{
    static void CheckCublasStatus(cublasStatus_t cublasStat, std::string msg = "CUBLAS error")
    {
        #ifdef __NVCC__
        if (cublasStat != CUBLAS_STATUS_SUCCESS)
        {               
            std::cout << msg;
            throw std::runtime_error(msg);
        }
        #else
        std::cout << "CublasHelper::CublasDestroy(): CUDA is not supported!" << std::endl;
        #endif
    }

    /// @brief Инициализирует CuBLAS
    /// @return 
    static cublasHandle_t CublasCreate()
    {
        #ifdef __NVCC__
        cublasHandle_t cublasH = nullptr;        
        cublasStatus_t cublasStat = cublasCreate(&cublasH);
        CublasHelper::CheckCublasStatus(cublasStat, "CUBLAS initialization failed\n");
        return cublasH;
        #else
        std::string msg{"CublasHelper::CublasCreate(): CUDA is not supported!"};
        std::cout << msg << std::endl;
        throw std::runtime_error(msg);
        #endif
    }

    /// @brief Инициализирует CuBLAS
    /// @return 
    static cublasHandle_t CublasCreate(int deviceId)
    {
        #ifdef __NVCC__
        cublasHandle_t cublasH = nullptr;        
        cublasStatus_t cublasStat;

        if(deviceId == 0)
        {
            cublasH = CublasCreate();
            return cublasH;
        }

        std::thread th{
            [&](){
                cudaSetDevice(deviceId);                
                cublasStat = cublasCreate(&cublasH);
            }
        };
        th.join();
        CublasHelper::CheckCublasStatus(cublasStat, "CUBLAS initialization failed\n");
        return cublasH;
        #else
        std::string msg{"CublasHelper::CublasCreate(int deviceId): CUDA is not supported!"};
        std::cout << msg << std::endl;
        throw std::runtime_error(msg);
        #endif
    }

    /// @brief Освобождает ресурсы CuBLAS
    /// @param cublasH 
    static void CublasDestroy(cublasHandle_t cublasH)
    {
        #ifdef __NVCC__
        cublasDestroy(cublasH);
        #else
        std::cout << "CublasHelper::CublasDestroy(): CUDA is not supported!" << std::endl;
        #endif
    }

    /// @brief Освобождает ресурсы CuBLAS
    /// @param cublasHandles 
    static void CublasDestroy(std::vector<cublasHandle_t> cublasHandles)
    {
        for (size_t i = 0; i < cublasHandles.size(); i++)
        {
            CublasDestroy(cublasHandles[i]);
        }
    }
};