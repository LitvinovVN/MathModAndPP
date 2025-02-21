#pragma once

#include <iostream>
#include "IVector.hpp"
#include "../CommonHelpers/DataLocation.hpp"
#include "../Arrays/DevMemArrPointer.hpp"
#include "../Arrays/DevMemArrPointers.hpp"

template<typename T>
class VectorRamGpus : IVector<T>
{
    // Массив указателей на части вектора, расположенные в различных областях памяти
    std::vector<DevMemArrPointer<T>> dataPointers;
    DevMemArrPointers<T> devMemArrPointers;

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

    VectorRamGpus()
    {
    }

    void InitByVal(T val) override
    {
        throw std::runtime_error("Not realized!");
        /*for (size_t i = 0; i < size; i++)
        {
            data[i] = val;
        }  */     
    }

    void Print() const override
    {
        std::cout << "VectorRamGpus::Print()" << std::endl;
        std::cout << this << std::endl;
        std::cout << "vectorType: " << vectorType << std::endl;
        std::cout << "dataPointers: ";
        if(dataPointers.size() == 0)
            std::cout << "none";
        for (size_t i = 0; i < dataPointers.size(); i++)
        {
            dataPointers[i].Print();
        }
        std::cout << std::endl;
    }

    /// @brief Выводит в консоль элементы вектора в заданном диапазоне
    void PrintData(unsigned long long indStart,
        unsigned long long length) const override
    {
        std::string elementSplitter = " ";
        if(vectorType == VectorType::VectorColumn)
            elementSplitter = "\n";

        // Глобальный индекс текущего элемента вектора
        unsigned long long i = indStart;
        // Глобальный индекс последнего выводимого в консоль элемента
        unsigned long long i_end = indStart + length - 1;
        // Определяем глобальный индекс первого элемента текущего блока
        unsigned long long dp_i_start = 0;
        // Определяем глобальный индекс последнего элемента текущего блока
        unsigned long long dp_i_end = 0;
        // Цикл перебора блоков
        for (size_t dp = 0; dp < dataPointers.size(); dp++)
        {
            auto& curDataPointer = dataPointers[dp];
            //auto data = curDataPointer.ptr;
            // Определяем глобальный индекс последнего элемента текущего блока
            dp_i_end += dataPointers[i].length - 1;
            // Если i лежит за пределами текущего блока - переходим к следующему
            if (i > dp_i_end) continue;
            // Смещаемся к нужному элементу текущего блока
            // Определяем локальный индекс элемента i в текущем блоке
            auto dp_i = i - dp_i_start;
            // Определяем количество элементов до конца блока
            //auto dp_i_to_end_length = dp_i_end - dp_i;
            // Определяем границу перебора
            auto i_last = i_end;
            if (i_last > dp_i_end)
                i_last = dp_i_end;

            /*for (unsigned long long cbi = cbi_s; cbi <= cbi_e; cbi++)
            {
                std::cout << data[i] << elementSplitter;
                i++;
            }*/
            // Смещаем глобальный индекс начала блока
            dp_i_start = dp_i_end + 1;
        }
        
    }

    size_t Size() const override
    {
        throw std::runtime_error("Not realized!");
        //return size;
    }

    /// @brief Возвращает значение элемента вектора, расположенного по указанному индексу
    T GetValue(unsigned long long index) const override
    {
        // Строим карту индексов
        // 0       999
        // 1000    1099 и т.д.
        std::vector<std::vector<unsigned long long>> indexMap;
        unsigned long long i = 0;
        for (size_t dp = 0; dp < dataPointers.size(); dp++)
        {
            std::vector<unsigned long long> row;
            row.push_back(i);
            i += dataPointers[dp].length - 1;
            row.push_back(i);

            indexMap.push_back(row);
        }
        

        throw std::runtime_error("Not realized!");
    }

    /// @brief Транспонирует вектор
    void Transpose()
    {        
        vectorType = (VectorType)!(bool)vectorType;
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
};