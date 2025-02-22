#pragma once

#include <iostream>
#include "IVector.hpp"
#include "../CommonHelpers/DataLocation.hpp"
#include "../Arrays/DevMemArrPointers.hpp"
#include "../Arrays/ArraysIndexMap.hpp"

template<typename T>
class VectorRamGpus : public IVector<T>
{
    // Контейнер указателей на части вектора, расположенные в различных областях памяти
    DevMemArrPointers<T> devMemArrPointers;
    
public:

    VectorRamGpus()
    {
    }

    void InitByVal(T val) override
    {
        throw std::runtime_error("Not realized!");
    }

    void Print() const override
    {
        std::cout << "VectorRamGpus::Print()" << std::endl;
        std::cout << this << std::endl;
        std::cout << "vectorType: " << this->vectorType << std::endl;
        std::cout << "devMemArrPointers: ";
        devMemArrPointers.Print();
        std::cout << std::endl;
    }

    /// @brief Выводит в консоль элементы вектора в заданном диапазоне
    void PrintData(unsigned long long indStart,
        unsigned long long length) const override
    {
        std::string elementSplitter = " ";
        if(this->vectorType == VectorType::VectorColumn)
            elementSplitter = "\n";

        // Глобальный индекс текущего элемента вектора
        unsigned long long i = indStart;
        // Глобальный индекс последнего выводимого в консоль элемента
        unsigned long long i_end = indStart + length - 1;
        
        std::cout << "Not realized!" << std::endl;
    }

    size_t Size() const override
    {
        throw std::runtime_error("Not realized!");
    }

    /// @brief Возвращает значение элемента вектора, расположенного по указанному индексу
    T GetValue(unsigned long long index) const override
    {
        T value = devMemArrPointers.GetValue(index);
        
        return value;
    }

    /// @brief Устанавливает значение элемента вектора, расположенного по указанному индексу
    T SetValue(unsigned long long index) const override
    {        
        throw std::runtime_error("Not realized!");
    }

    /// @brief Транспонирует вектор
    void Transpose()
    {        
        this->vectorType = (VectorType)!(bool)this->vectorType;
    }

    ///// Выделение блоков памяти /////
    
    /// @brief Выделяет непрерывный блок памяти
    /// @param id Идентификатор блока (>0)
    /// @param dataLocation Место расположения блока памяти 
    /// @param length Количество элементов в блоке
    /// @return DevMemArrPointer
    /*DevMemArrPointer<T> AllocMem(unsigned id,
        DataLocation dataLocation,
        unsigned long long length)
    {
        if (id==0)
            return DevMemArrPointer<T>{};

        auto dmptr = devMemArrPointers.AllocMem(id, dataLocation, length);

        return dmptr;
    }*/

    /// @brief Добавляет элементы в вектор
    /// @param dataLocation Место расположения элементов вектора
    /// @param length Количество добавляемых элементов
    /// @return bool - Результат выполнения операции (true - успех)
    bool Add(DataLocation dataLocation,
        unsigned long long length)
    {
        auto result = devMemArrPointers.AddBlock(dataLocation, length);
        
        return result;
    }
    ///////////////////////////////////

    
    /// @brief Освобождает всю зарезервированную память
    void Clear()
    {
        devMemArrPointers.Clear();
    }

    ///////////////////////////////////
};