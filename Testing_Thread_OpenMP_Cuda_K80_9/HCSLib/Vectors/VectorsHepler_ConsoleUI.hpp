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
        //std::cout << "v1.Print();" << std::endl;
        //v1.Print();

        std::cout << "auto res = v1.AllocMem(1, DataLocation::RAM, 100000000);" << std::endl;
        auto res = v1.Add(DataLocation::RAM, 100000000);
        std::cout << "res: " << res << std::endl;
        std::cout << "v1.Print();" << std::endl;
        v1.Print();
        std::cout << std::endl;
        
        std::cout << "res = v1.AllocMem(2, DataLocation::GPU0, 100000000);" << std::endl;
        res = v1.Add(DataLocation::GPU0, 100000000);
        std::cout << "res: " << res << std::endl;
        std::cout << "v1.Print();" << std::endl;
        v1.Print();
        std::cout << std::endl;

        /*std::cout << "v1.Transpose();" << std::endl;
        v1.Transpose();
        std::cout << "v1.Print();" << std::endl;
        v1.Print();
        std::cout << "v1.PrintData(0, 15);" << std::endl;
        v1.PrintData(0, 15);
        std::cout << std::endl;

        std::cout << "v1.Transpose();" << std::endl;
        v1.Transpose();
        std::cout << "v1.Print();" << std::endl;
        v1.Print();
        std::cout << "v1.PrintData(0, 15);" << std::endl;
        v1.PrintData(0, 15);
        std::cout << std::endl;*/

        ConsoleHelper::PrintLine("bool isValueSetted = v1.SetValue(99999999, 123.45);");
        bool isValueSetted = v1.SetValue(99999999, 123.45);
        ConsoleHelper::PrintKeyValue("isValueSetted", isValueSetted);

        ConsoleHelper::PrintLine("auto val99999999 = v1.GetValue(99999999);");
        auto val99999999 = v1.GetValue(99999999);
        ConsoleHelper::PrintKeyValue("val99999999", val99999999);

        std::cout << "bool isValueSetted100M = v1.SetValue(100000000, 23.455);" << std::endl;
        bool isValueSetted100M = v1.SetValue(100000000, 23.455);
        std::cout << "isValueSetted100M: " << isValueSetted100M << std::endl;

        ConsoleHelper::PrintLine("v1.GetValue(100000000);");
        auto val100M = v1.GetValue(100000000);
        ConsoleHelper::PrintKeyValue("val100M", val100M);

        ConsoleHelper::PrintLine("bool isValueSetted100M1 = v1.SetValue(100000001, 23.456);");
        bool isValueSetted100M1 = v1.SetValue(100000001, 23.456);
        ConsoleHelper::PrintKeyValue("isValueSetted100M1", isValueSetted100M1);

        std::cout << "v1.GetValue(100000001);" << std::endl;
        auto val100M1 = v1.GetValue(100000001);
        std::cout << "val100M1: " << val100M1 << std::endl;

        ConsoleHelper::PrintLine("v1.PrintData(99999999, 3);");
        v1.PrintData(99999999, 3);

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