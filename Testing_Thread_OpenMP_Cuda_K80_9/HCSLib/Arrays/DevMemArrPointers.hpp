#pragma once

#include <iostream>
#include "../CommonHelpers/DataLocation.hpp"
#include "ArrayHelper.hpp"
#include "DevMemArrPointer.hpp"

template<typename T>
class DevMemArrPointers
{
    // Массив указателей на части вектора, расположенные в различных областях памяти
    std::vector<DevMemArrPointer<T>> dataPointers;

    /// @brief Очищает dataPointers от сброшенных в исходное состояние объектов DevMemArrPointer<T>
    void RemoveFreeDataPointers()
    {
        bool isClean = false;
        while(!isClean)
        {
            isClean = true;
            for (size_t i = 0; i < dataPointers.size(); i++)
            {
                if(!dataPointers[i].IsReset())
                {
                    isClean = false;
                    dataPointers.erase(dataPointers.begin() + i);
                    break;
                }
            }
            
        }
    }

public:

    DevMemArrPointers()
    {
    }

    void InitByVal(T val)
    {
        throw std::runtime_error("Not realized!");
        /*for (size_t i = 0; i < size; i++)
        {
            data[i] = val;
        }  */     
    }

    void Print() const
    {
        std::cout << "DevMemArrPointers::Print()" << std::endl;

        std::cout << "dataPointers: ";
        if(dataPointers.size() == 0)
            std::cout << "none";
        for (size_t i = 0; i < dataPointers.size(); i++)
        {
            dataPointers[i].Print();
        }
        std::cout << std::endl;
    }
    
    size_t Size() const
    {
        throw std::runtime_error("Not realized!");
        //return size;
    }
    
    ///// Выделение блоков памяти /////
    
    /// @brief Выделяет непрерывный блок памяти
    /// @param id Идентификатор блока (>0)
    /// @param dataLocation Место расположения блока памяти 
    /// @param length Количество элементов в блоке
    /// @return DevMemArrPointer
    DevMemArrPointer<T> AllocMem(unsigned id,
        DataLocation dataLocation,
        unsigned long long length)
    {
        if (id==0)
            return DevMemArrPointer<T>{};

        T* ptr = nullptr;

        try
        {
            switch (dataLocation)
            {
            case DataLocation::RAM:
                ptr = ArrayHelper::CreateArrayRam<T>(length);
                break;
            case DataLocation::GPU0:
                ptr = ArrayHelper::CreateArrayGpu<T>(length, 0);
                break;
            case DataLocation::GPU1:
                ptr = ArrayHelper::CreateArrayGpu<T>(length, 1);
                break;
            case DataLocation::GPU2:
                ptr = ArrayHelper::CreateArrayGpu<T>(length, 2);
                break;
            case DataLocation::GPU3:
                ptr = ArrayHelper::CreateArrayGpu<T>(length, 3);
                break;
            
            default:
                break;
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return DevMemArrPointer<T>{};
        }
        
        DevMemArrPointer<T> dmptr(id, dataLocation, ptr, length);
        dataPointers.push_back(dmptr);

        return dmptr;
    }
    ///////////////////////////////////

    ///// Освобождение памяти /////

    /// @brief Освобождает зарезервированную память
    void Clear(DevMemArrPointer<T>& devMemArrPointer)
    {
        std::cout << "Clear ";
        devMemArrPointer.Print();
        std::cout << std::endl;
        try
        {
            switch (devMemArrPointer.dataLocation)
            {
            case DataLocation::RAM:
                ArrayHelper::DeleteArrayRam<T>(devMemArrPointer.ptr);
                break;
            case DataLocation::GPU0:
                ArrayHelper::DeleteArrayGpu<T>(devMemArrPointer.ptr, 0);
                break;
            case DataLocation::GPU1:
                ArrayHelper::DeleteArrayGpu<T>(devMemArrPointer.ptr, 1);
                break;
            case DataLocation::GPU2:
                ArrayHelper::DeleteArrayGpu<T>(devMemArrPointer.ptr, 2);
                break;
            case DataLocation::GPU3:
                ArrayHelper::DeleteArrayGpu<T>(devMemArrPointer.ptr, 3);
                break;
            
            default:
                break;
            }

            if(!devMemArrPointer.ptr)
            {
                devMemArrPointer.Reset();
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    /// @brief Освобождает всю зарезервированную память
    void Clear()
    {
        // Очищаем зарезервированную память
        for(auto& dataPointer : dataPointers)
        {
            Clear(dataPointer);
        }
        // Очищаем контейнер dataPointers
        RemoveFreeDataPointers();
    }

    ///////////////////////////////////

    /// @brief Строит карту индексов
    /// @return std::vector<std::vector<unsigned long long>>
    ArraysIndexMap GetArraysIndexMap() const
    {
        // Строим карту индексов
        // 0       999
        // 1000    1099 и т.д.
        ArraysIndexMap indexMap;
        unsigned long long i = 0;
        for (size_t dp = 0; dp < dataPointers.size(); dp++)
        {
            auto indStart = i;
            i += dataPointers[dp].length - 1;
            auto indEnd = i;

            indexMap.AddIndexes(indStart, indEnd);

            i++;
        }

        return indexMap;
    }
        
};