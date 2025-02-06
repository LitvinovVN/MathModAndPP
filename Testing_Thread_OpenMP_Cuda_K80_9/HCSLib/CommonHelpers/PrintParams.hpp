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
};