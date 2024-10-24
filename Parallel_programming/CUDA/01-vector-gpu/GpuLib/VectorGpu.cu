#pragma once

#include <iostream>
#include "CudaHelper.cu"
#include <chrono>

/// @brief Структура для хранения результата запуска 
template<typename T = double>
struct FuncResultScalar
{
    // Статус выполнения функции
    bool Status = true;
    // Результат функции
    T Result;
    // Время выполнения функции, мкс
    long long Time_mks;

    void Print()
    {
        std::cout << "["   << Status 
                  << ", "  << Result
                  << ", "  << Time_mks
                  << " mks]"   << std::endl;
    }    
};

std::ostream& operator<<(std::ostream &stream, const FuncResultScalar<> &funcResultScalar)
{
    return stream   << "["       << funcResultScalar.Status 
                    << ", "      << funcResultScalar.Result
                    << ", "      << funcResultScalar.Time_mks
                    << " mks]";
}

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
        //std::cout << "VectorGpu(size_t size) constructor started...\n";

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

        //std::cout << "VectorGpu(size_t size): Device memory for VectorGpu allocated!\n";
    }

    ~VectorGpu()
    {
        //std::cout << "~VectorGpu(): " << this << " destructed!\n";
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
    FuncResultScalar<T> Sum(unsigned blocksNum, unsigned threadsNum)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        if(!CheckState())
            throw std::logic_error("Vector is not initialized!");      

        T result = CudaHelper<T>::Sum(_dev_data, _size, blocksNum, threadsNum);       

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

        FuncResultScalar<T> res{true, result, std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()};
        //res.Print();

        return res;
    }

    /// @brief Освобождаем массив в видеопамяти
    void Clear_dev_data()
    {
        if(_dev_data != nullptr)
        {
            cudaFree(_dev_data);
            _dev_data = nullptr;
            _isInitialized = false;
            //std::cout << "Device memory for VectorGpu cleared!\n";
        }
    }

    /// @brief Возвращает указатель на данные в видеопамяти
    /// @return 
    __host__ __device__
    T* Get_dev_data_pointer()
    {
        return _dev_data;
    }

    __host__ __device__
    size_t GetSize() const
    {
        return _size;
    }
    
    /// @brief Инициализирует вектор числом
    void InitVectorByScalar(double value)
    {
        // Создаём временный массив
        T* tmp = new T[_size];
        
        // Инициализируем временный массив        
        for (auto i = 0; i < _size; i++)
        {
            tmp[i] = value;
            //std::cout << tmp[i] << " ";
        }
        //std::cout << std::endl;

        // Копируем данные из временного массива в видеопамять
        cudaError_t cudaResult = cudaMemcpy(_dev_data, tmp, _size*sizeof(T), cudaMemcpyHostToDevice);
        if (cudaResult != cudaSuccess)
        {
            std::string msg("Could not copy data from RAM to device memory: ");
            msg += cudaGetErrorString(cudaResult);
            throw std::runtime_error(msg);
        }

        //std::cout << "cudaMemCpy OK!\n";

        // Освобождаем временный массив
        delete[] tmp;

        // Устанавливаем флаг инициализации вектора
        _isInitialized = true;
    }

    /// @brief Инициализирует вектор числами из диапазона от start до end
    void InitVectorByRange(double start, double end)
    {
        // Создаём временный массив
        T* tmp = new T[_size];
        size_t cnt = 0;

        // Инициализируем временный массив
        auto step = (end-start)/(_size-1);
        for (auto i = start; i < end+step/2; i+=step)
        {
            tmp[cnt++] = i;
            //std::cout << tmp[cnt-1] << " ";
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

        //std::cout << "cudaMemCpy OK!\n";

        // Освобождаем временный массив
        delete[] tmp;

        // Устанавливаем флаг инициализации вектора
        _isInitialized = true;
    }

};