#pragma once

#include <iostream>
#include "../CommonHelpers/DataLocation.hpp"
#include "ArrayHelper.hpp"
#include "DevMemArrPointer.hpp"
#include "ArrayBlockIndexes.hpp"

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

    void InitByVal(T value)
    {        
        for (auto& devMemArrPointer : dataPointers)
        {
            if(!devMemArrPointer.IsInitialized())
                continue;

            ArrayHelper::InitArray(devMemArrPointer, value);
        }
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
    
    size_t GetSize() const
    {
        unsigned long long size = 0;
        for (auto& devMemArrPointer : dataPointers)
        {
            size += devMemArrPointer.length;
        }

        return size;
    }
    
    /// @brief Возвращает количество выделенных блоков памяти
    /// @return size_t
    auto GetDataPointersNum()
    {
        return dataPointers.size();
    }

    ///// Выделение блоков памяти /////
    
    /// @brief Выделяет непрерывный блок памяти
    /// @param id Идентификатор блока
    /// @param dataLocation Место расположения блока данных 
    /// @param length Количество элементов в блоке
    /// @return DevMemArrPointer
    DevMemArrPointer<T> AllocMem(unsigned id,
        DataLocation dataLocation,
        unsigned long long length)
    {
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
        
        if(!ptr)
            return DevMemArrPointer<T>{};

        DevMemArrPointer<T> dmptr(id, dataLocation, ptr, length);
        dataPointers.push_back(dmptr);
        
        return dmptr;
    }
    
    /// @brief Добавляет непрерывный блок данных
    /// @param dataLocation Место расположения блока данных
    /// @param length Количество элементов в блоке
    /// @return bool - Результат выполнения операции (true - успех)
    bool AddBlock(DataLocation dataLocation,
        unsigned long long length)
    {
        auto newBlockId = GetDataPointersNum();
        auto newBlock = AllocMem(newBlockId, dataLocation, length);
        if(newBlock.IsInitialized())
            return true;
        
        return false;
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

    
    /// @brief Возвращает значение по глобальному индексу
    /// @param globalIndex Глобальный индекс
    /// @return 
    T GetValue(unsigned long long globalIndex) const
    {
        ArraysIndexMap map = GetArraysIndexMap();
        //map.Print();
        // Определяем индекс блока и локальный индекс элемента в блоке
        ArrayBlockIndexes indexes = map.GetArrayBlockIndexes(globalIndex);
        //std::cout << "globalIndex: [" << globalIndex << "]: ";
        //indexes.Print();

        if(!indexes.IsInitialized())
            throw std::runtime_error("Error in finding ArrayBlockIndexes by globalIndex!");

        T value = GetValue(indexes.blockIndex, indexes.localIndex);

        return value;
    }

    /// @brief Возвращает значение по индексу блока и локальному индексу
    /// @param blockIndex 
    /// @param localIndex 
    /// @return 
    T GetValue(unsigned blockIndex, unsigned long long localIndex) const
    {
        auto& devMemArrPointer = dataPointers[blockIndex];
        T value;
        switch (devMemArrPointer.dataLocation)
        {
        case DataLocation::RAM:
            value = ArrayHelper::GetValueRAM(devMemArrPointer.ptr, localIndex);
            break;
        case DataLocation::GPU0:
            value = ArrayHelper::GetValueGPU(devMemArrPointer.ptr, localIndex, 0);
            break;
        case DataLocation::GPU1:
            value = ArrayHelper::GetValueGPU(devMemArrPointer.ptr, localIndex, 1);
            break;
        case DataLocation::GPU2:
            value = ArrayHelper::GetValueGPU(devMemArrPointer.ptr, localIndex, 2);
            break;
        case DataLocation::GPU3:
            value = ArrayHelper::GetValueGPU(devMemArrPointer.ptr, localIndex, 3);
            break;
        
        default:
            throw std::runtime_error("Wrong DataLocation!");            
        }

        return value;
    }

    /// @brief Устанавливает значение по глобальному индексу
    /// @param globalIndex 
    /// @param value 
    /// @return 
    bool SetValue(unsigned long long globalIndex, T value)
    {
        ArraysIndexMap map = GetArraysIndexMap();
        //map.Print();
        // Определяем индекс блока и локальный индекс элемента в блоке
        ArrayBlockIndexes indexes = map.GetArrayBlockIndexes(globalIndex);
        //std::cout << "globalIndex: [" << globalIndex << "]: ";
        //indexes.Print();

        if(!indexes.IsInitialized())
            return false;

        bool isValueSetted = SetValue(indexes.blockIndex, indexes.localIndex, value);
        return isValueSetted;
    }

    bool SetValue(unsigned blockIndex, unsigned long long localIndex, T value)
    {
        try
        {
            auto& devMemArrPointer = dataPointers[blockIndex];
        
            switch (devMemArrPointer.dataLocation)
            {
            case DataLocation::RAM:
                ArrayHelper::SetValueRAM(devMemArrPointer.ptr, localIndex, value);
                break;
            case DataLocation::GPU0:
                ArrayHelper::SetValueGPU(devMemArrPointer.ptr, localIndex, 0, value);
                break;
            case DataLocation::GPU1:
                ArrayHelper::SetValueGPU(devMemArrPointer.ptr, localIndex, 1, value);
                break;
            case DataLocation::GPU2:
                ArrayHelper::SetValueGPU(devMemArrPointer.ptr, localIndex, 2, value);
                break;
            case DataLocation::GPU3:
                ArrayHelper::SetValueGPU(devMemArrPointer.ptr, localIndex, 3, value);
                break;
            
            default:
                throw std::runtime_error("Wrong DataLocation!");                
            }

            return true;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return false;
        }
    
    }

    template<typename S>
    void Multiply(S scalar, bool isParallel = false)
    {
        if(!isParallel)
        {
            for (auto devMemArrPointer : dataPointers)
            {
                ArrayHelper::Multiply(devMemArrPointer, scalar);
            }
        }
        else
        {
            std::vector<std::thread> threads;
            for (auto devMemArrPointer : dataPointers)
            {
                threads.push_back(
                    std::thread{
                        [=](){
                            ArrayHelper::MultiplyParallel(devMemArrPointer, scalar, 10);
                        }
                    }
                );
            }

            for(auto& th : threads)
            {
                if(th.joinable())
                    th.join();
            }
        }
    }


};