#pragma once

#include <iostream>
#include <map>
#include "XYInd.hpp"
#include "ZMask.hpp"

struct ZMaskRepository
{
    std::map<XYInd, ZMask> data;
    
    void Print() const
    {
        std::cout << "ZMaskRepository:" << std::endl;
        for (auto it = data.begin(); it != data.end(); ++it)
        {
            std::cout << "(" << it->first.x << ", " << it->first.y << ")" << std::endl;
            it->second.Print();
        }
        std::cout << "-------------------" << std::endl;
    }
};
