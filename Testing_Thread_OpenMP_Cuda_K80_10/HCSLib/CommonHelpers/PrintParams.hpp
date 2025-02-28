#pragma once

#include <iostream>

struct PrintParams
{
    std::string startMes = "[";
    std::string splitterKeyValue = ": ";
    std::string splitter = "; ";
    std::string endMes = "]";
    bool isEndl = true;

    PrintParams& SetIsEndl(bool isEndLine = true)
    {
        isEndl = isEndLine;
        return *this;
    }

    void PrintStartMessage()
    {
        std::cout << startMes;
    }

    void PrintEndMessage()
    {
        std::cout << endMes;
    }

    void PrintSplitter()
    {
        std::cout << splitter;
    }

    void PrintKeyValue(std::string key, unsigned value)
    {
        std::cout << key << splitterKeyValue << value;
    }

    void PrintIsEndl()
    {
        if (isEndl)
            std::cout << std::endl;
    }
        
};