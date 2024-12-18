#pragma once

#include <iostream>

struct PrintParams
{
    std::string startMes = "[";
    std::string splitterKeyValue = ": ";
    std::string splitter = "; ";
    std::string endMes = "]";
    bool isEndl = true;
};