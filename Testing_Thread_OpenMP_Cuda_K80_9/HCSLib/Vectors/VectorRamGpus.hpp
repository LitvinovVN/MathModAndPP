#pragma once

#include <iostream>
#include "IVector.hpp"

template<typename T>
class VectorRamGpus : IVector<T>
{
    // Размеры частей вектора, расположенных в различных видах памяти
    // 0 - RAM
    // 1 - GPU0
    // 2 - GPU1
    // и т.д.
    std::vector<unsigned long> sizes;

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
        throw std::runtime_error("Not realized!");
        /*for (size_t i = 0; i < size; i++)
        {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;    */ 
    }

    size_t Size() const override
    {
        throw std::runtime_error("Not realized!");
        //return size;
    }

    void Print()
    {
        std::cout << "VectorRamGpus::Print()" << std::endl;
        std::cout << this << std::endl;
        std::cout << "sizes: ";
        for (size_t i = 0; i < sizes.size(); i++)
        {
            std::cout << sizes[i] << " ";
        }
        std::cout << std::endl;
    }
};