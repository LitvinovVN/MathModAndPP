#pragma once

#include <iostream>

#include "VectorRamGpusHelper.hpp"
#include "../CommonHelpers/DataLocation.hpp"

struct VectorsHelper_ConsoleUI
{

    static void VectorRamGpus_ConsoleUI()
    {
        std::cout << "VectorRamGpus_ConsoleUI" << std::endl;

        ConsoleHelper::PrintLine("VectorRamGpus<double> v1;");
        VectorRamGpus<double> v1;
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("auto res = v1.Add(DataLocation::RAM, 10);");
        auto res = v1.Add(DataLocation::RAM, 10);
        ConsoleHelper::PrintKeyValue("res", res);
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        ConsoleHelper::PrintLine("----------\n");
        
        ConsoleHelper::PrintLine("res = v1.Add(DataLocation::GPU0, 5);");
        res = v1.Add(DataLocation::GPU0, 5);
        ConsoleHelper::PrintKeyValue("res", res);
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("res = v1.Add(DataLocation::GPU1, 7);");
        res = v1.Add(DataLocation::GPU1, 7);
        ConsoleHelper::PrintKeyValue("res", res);
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("auto size = v1.Size();");
        auto size = v1.Size();
        ConsoleHelper::PrintKeyValue("size", size);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.PrintData(0, 22);");
        v1.PrintData(0, 22);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.InitByVal(0.01);");
        v1.InitByVal(0.01);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.PrintData(0, 22);");
        v1.PrintData(0, 22);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.Transpose();");
        v1.Transpose();
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        std::cout << "v1.PrintData(0, 5);" << std::endl;
        v1.PrintData(0, 5);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.Transpose();");
        v1.Transpose();
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        ConsoleHelper::PrintLine("v1.PrintData(0, 5);");
        v1.PrintData(0, 5);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("bool isValueSetted = v1.SetValue(1, 123.45);");
        bool isValueSetted = v1.SetValue(1, 123.45);
        ConsoleHelper::PrintKeyValue("isValueSetted", isValueSetted);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("auto 1 = v1.GetValue(1);");
        auto val = v1.GetValue(1);
        ConsoleHelper::PrintKeyValue("val", val);
        ConsoleHelper::PrintLine("----------\n");

        std::cout << "bool isValueSetted = v1.SetValue(11, 23.455);" << std::endl;
        isValueSetted = v1.SetValue(11, 23.455);
        ConsoleHelper::PrintKeyValue("isValueSetted", isValueSetted);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.GetValue(11);");
        val = v1.GetValue(11);
        ConsoleHelper::PrintKeyValue("val", val);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("bool isValueSetted = v1.SetValue(21, 23.456);");
        isValueSetted = v1.SetValue(21, 23.456);
        ConsoleHelper::PrintKeyValue("isValueSetted", isValueSetted);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("val = v1.GetValue(21);");
        val = v1.GetValue(21);
        ConsoleHelper::PrintKeyValue("val", val);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.PrintData(0, 22);");
        v1.PrintData(0, 22);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.Multiply(2);");
        v1.Multiply(2);
        ConsoleHelper::PrintLine("v1.PrintData(0, 22);");
        v1.PrintData(0, 22);
        ConsoleHelper::PrintLine("----------\n");

        bool isClear = ConsoleHelper::GetBoolFromUser("Do you want clear vector data? (y/n)");
        if(isClear)
        {
            ConsoleHelper::PrintLine("v1.Clear();");
            v1.Clear();
        }
        
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
    }

};