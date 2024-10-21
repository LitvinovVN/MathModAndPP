#pragma once

#include <iostream>
#include "CudaHelper.cu"

/// @brief Вектор (в GPU) 
/// @tparam T Тип элементов вектора
template<typename T = double>
class VectorGpu
{
    // Количество элементов вектора
    size_t _size = 0;
    // Указатель на массив в видеопамяти
    T* _dev_data = nullptr;
    // Флаг инициализации вектора
    // false - неинициализирован, true - инициализирован
    bool _isInitialized = false;

public:
    VectorGpu(size_t size) : _size(size)
    {
        std::cout << "VectorGpu(size_t size) constructor started...\n";

        if (_size == 0)
        {
            std::string mes = "Cannot initialize vector of _size = 0";
            //std::cerr << mes << std::endl;
            throw std::logic_error(mes);
        }

        cudaError_t cudaResult = cudaMalloc(&_dev_data, size*sizeof(T));
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not allocate device memory for VectorGpu: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }

        std::cout << "Device memory for VectorGpu allocated!\n";
    }

    ~VectorGpu()
    {
        std::cout << "~VectorGpu(): " << this << " destructed!\n";
    }

    /// @brief Проверяет состояние вектора
    bool CheckState()
    {
        if(!_isInitialized)
            return false;

        if(_size < 1)
            return false;

        if(_dev_data == nullptr)
            return false;

        return true;
    }

    /// @brief Возвращает сумму элементов вектора
    T Sum(unsigned blocksNum, unsigned threadsNum)
    {
        if(!CheckState())
            throw std::logic_error("Vector is not initialized!");      

        T result = CudaHelper<T>::Sum(_dev_data, _size, blocksNum, threadsNum);

        return result;
    }

    /// @brief Освобождаем массив в видеопамяти
    void Clear_dev_data()
    {
        if(_dev_data != nullptr)
        {
            cudaFree(_dev_data);
            _dev_data = nullptr;
            _isInitialized = false;
            std::cout << "Device memory for VectorGpu cleared!\n";
        }
    }

    /// @brief Возвращает указатель на данные в видеопамяти
    /// @return 
    __host__ __device__
    T* get_dev_data_pointer()
    {
        return _dev_data;
    }

    __host__ __device__
    size_t getSize() const
    {
        return _size;
    }
    
    void initVectorByRange(double start, double end)
    {
        // Создаём временный массив
        T* tmp = new T[_size];
        size_t cnt = 0;

        // Инициализируем временный массив
        auto step = (end-start)/(_size-1);
        for (auto i = start; i < end+step/2; i+=step)
        {
            tmp[cnt++] = i;
            std::cout << tmp[cnt-1] << " ";
        }
        std::cout << std::endl;

        // Копируем данные из временного массива в видеопамять
        cudaError_t cudaResult = cudaMemcpy(_dev_data, tmp, _size*sizeof(T), cudaMemcpyHostToDevice);
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not copy data from RAM to device memory: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }

        std::cout << "cudaMemCpy OK!\n";

        // Освобождаем временный массив
        delete[] tmp;

        // Устанавливаем флаг инициализации вектора
        _isInitialized = true;
    }

};