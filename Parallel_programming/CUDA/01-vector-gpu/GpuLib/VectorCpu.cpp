#pragma once

#include <iostream>
#include <chrono>
#include "FuncResultScalar.cpp"

/// @brief Вектор (в RAM) 
/// @tparam T Тип элементов вектора
template<typename T = double>
class VectorCpu
{
    // Количество элементов вектора
    size_t _size = 0;
    // Указатель на массив в RAM
    T* _data = nullptr;
    // Флаг инициализации вектора
    // false - неинициализирован, true - инициализирован
    bool _isInitialized = false;

public:
    VectorCpu(size_t size) : _size(size)
    {
        //std::cout << "VectorCpu(size_t size) constructor started...\n";

        if (_size == 0)
        {
            std::string mes = "Cannot initialize vector of _size = 0";
            //std::cerr << mes << std::endl;
            throw std::logic_error(mes);
        }

        _data = new T[_size];

        //std::cout << "VectorCpu(size_t size): Device memory for VectorGpu allocated!\n";
    }

    ~VectorCpu()
    {
        //std::cout << "~VectorCpu(): " << this << " destructed!\n";
    }

    /// @brief Проверяет состояние вектора
    bool CheckState()
    {
        if(!_isInitialized)
            return false;

        if(_size < 1)
            return false;

        if(_data == nullptr)
            return false;

        return true;
    }

    /// @brief Возвращает сумму элементов вектора
    FuncResultScalar<T> Sum(unsigned threadsNum)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        if(!CheckState())
            throw std::logic_error("Vector is not initialized!");      

        T result = RamArrayHelper<T>::Sum(_data, _size, threadsNum); 
        //T result = (T)0;      

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

        FuncResultScalar<T> res{true, result, std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()};
        //res.Print();

        return res;
    }

    /// @brief Освобождаем массив в RAM
    void Clear_data()
    {
        if(_data != nullptr)
        {
            delete[] _data;
            _data = nullptr;
            _isInitialized = false;
            //std::cout << "RAM for VectorCpu cleared!\n";
        }
    }

    /// @brief Возвращает указатель на данные    
    T* Get_data_pointer()
    {
        return _data;
    }
    
    /// @brief Возвращает размер вектора
    size_t GetSize() const
    {
        return _size;
    }
    
    /// @brief Инициализирует вектор числом
    void InitVectorByScalar(double value)
    {               
        for (auto i = 0; i < _size; i++)
        {
            _data[i] = value;            
        }        

        // Устанавливаем флаг инициализации вектора
        _isInitialized = true;
    }

    /// @brief Инициализирует вектор числами из диапазона от start до end
    void InitVectorByRange(double start, double end)
    {
        size_t cnt = 0;        
        auto step = (end-start)/(_size-1);
        for (auto i = start; i < end+step/2; i+=step)
        {
            _data[cnt++] = i;
        }       

        // Устанавливаем флаг инициализации вектора
        _isInitialized = true;
    }

    void Print()
    {
        if(!_isInitialized)
        {
            std::cout << "VectorCpu(" << _size << ") is not initialized!" << std::endl;
            return;
        }

        std::cout << "[";
        for(size_t i{0}; i < _size; i++)
            std::cout << _data[i] << " ";
        std::cout << "]" << std::endl;        
    }
};