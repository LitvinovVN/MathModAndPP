#pragma once

#include <iostream>

#include "VectorRamGpusHelper.hpp"

struct VectorsHelper_ConsoleUI
{

    static void VectorRamGpus_ConsoleUI()
    {
        std::cout << "VectorRamGpus_ConsoleUI" << std::endl;

        VectorRamGpus<double> v1;
        v1.Print();
    }

};