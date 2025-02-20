#pragma once

#include <iostream>
#include "DataLocation.hpp"
#include "PrintParams.hpp"

/// @brief Указатель на массив, расположенный в памяти устройства вычислительного узла (RAM или GPU)
template<typename T>
struct DevMemArrPointer
{
    /// Идентификатор указателя
    unsigned id = 0;
    // Место расположения данных
    DataLocation dataLocation = DataLocation::None;
    // Указатель на массив
    T* ptr = nullptr;
    // Количество элементов
    unsigned long long length = 0;  


    DevMemArrPointer()
    {}

    DevMemArrPointer(unsigned id,
        DataLocation dataLocation,
        T* ptr,
        unsigned long long length)
        : id(id), dataLocation(dataLocation),
          ptr(ptr), length(length)
    {
        if(id == 0)
            throw std::runtime_error("id must be > 0!");
    }


    /// @brief Возвращает флаг инициализации указателя
    /// @return 
    bool IsInitialized() const
    {
        if(    id           == 0
            || dataLocation == DataLocation::None
            || ptr          == nullptr
            || length       == 0)
        {
            return false;
        }

        return true;
    }

    /// @brief Возвращает флаг сброшенности указателя
    /// @return 
    bool IsReset() const
    {
        if(    id           == 0
            && dataLocation == DataLocation::None
            && ptr          == nullptr
            && length       == 0)
        {
            return false;
        }

        return true;
    }

    /// @brief Сбрасывает указатель в исходное неинициализированное состояние
    void Reset()
    {
        id           = 0;
        dataLocation = DataLocation::None;
        ptr          = nullptr;
        length       = 0;
    }

    /// @brief Возвращает объём памяти, занимаемый структурой
    /// @return unsigned long long (объём в байтах)
    unsigned long long GetSizeStruct() const
    {
        return sizeof(*this);
    }

    /// @brief Возвращает объём памяти, занимаемый массивом
    /// @return 
    unsigned long long GetSizeData() const
    {
        return sizeof(T) * length;
    }

    /// @brief Выводит в консоль сведения об указателе
    /// @param pp 
    void Print(PrintParams pp = PrintParams{}) const
    {
        std::cout << pp.startMes;
        std::cout << "id" << pp.splitterKeyValue << id;
        std::cout << pp.splitter;
        std::cout << "dataLocation" << pp.splitterKeyValue << dataLocation;
        std::cout << pp.splitter;
        std::cout << "ptr" << pp.splitterKeyValue << ptr;
        std::cout << pp.splitter;
        std::cout << "length" << pp.splitterKeyValue << length;
        std::cout << pp.splitter;
        std::cout << "GetSizeStruct()" << pp.splitterKeyValue << GetSizeStruct();
        std::cout << pp.splitter;
        std::cout << "GetSizeData()" << pp.splitterKeyValue << GetSizeData();
        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};