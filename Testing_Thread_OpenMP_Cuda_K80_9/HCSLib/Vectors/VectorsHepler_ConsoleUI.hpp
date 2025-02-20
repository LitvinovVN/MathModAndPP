#pragma once

#include <iostream>

#include "VectorRamGpusHelper.hpp"
#include "../CommonHelpers/DataLocation.hpp"

struct VectorsHelper_ConsoleUI
{

    static void VectorRamGpus_ConsoleUI()
    {
        std::cout << "VectorRamGpus_ConsoleUI" << std::endl;

        std::cout << "VectorRamGpus<double> v1;" << std::endl;
        VectorRamGpus<double> v1;
        std::cout << "v1.Print();" << std::endl;
        v1.Print();

        std::cout << "auto res = v1.AllocMem(1, DataLocation::RAM, 100000000);" << std::endl;
        auto res = v1.AllocMem(1, DataLocation::RAM, 100000000);
        std::cout << "res.Print();" << std::endl;
        res.Print();
        
        std::cout << "v1.Print();" << std::endl;
        v1.Print();

        std::cout << "res = v1.AllocMem(2, DataLocation::GPU0, 100000000);" << std::endl;
        res = v1.AllocMem(2, DataLocation::GPU0, 100000000);
        std::cout << "res.Print();" << std::endl;
        res.Print();
        
        std::cout << "v1.Print();" << std::endl;
        v1.Print();

        bool isClear = ConsoleHelper::GetBoolFromUser("Do you want clear vector data? (y/n)");
        if(isClear)
        {
            std::cout << "v1.Clear();" << std::endl;
            v1.Clear();
        }
        
        std::cout << "v1.Print();" << std::endl;
        v1.Print();
    }

};