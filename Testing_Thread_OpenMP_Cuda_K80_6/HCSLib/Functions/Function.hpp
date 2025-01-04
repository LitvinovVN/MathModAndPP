#pragma once

#include <iostream>
#include "../CommonHelpers/PrintParams.hpp"

class Function
{
    // Указатель на функцию, реализующую алгоритм
    void* func = nullptr;

public:
    void Print(PrintParams pp)
    {
        std::cout << pp.startMes;
        
        std::cout << "func" << pp.splitterKeyValue << func;
        //std::cout << pp.splitter;        
        
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }
};