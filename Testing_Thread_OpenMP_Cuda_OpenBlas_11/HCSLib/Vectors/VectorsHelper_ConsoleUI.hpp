#pragma once

#include <iostream>

#include "_IncludeVectors.hpp"
#include "../CommonHelpers/DataLocation.hpp"

struct VectorsHelper_ConsoleUI
{
    static void VectorRam_Console_UI()
    {
        std::cout << "VectorRam_Console_UI" << std::endl;

        ConsoleHelper::PrintLine("Creating VectorRam object with 10 elements");
        VectorRam<double> v1(10);
        v1.Print();
        v1.InitByVal(5);
        std::cout << "v1.SetValue(1, -10.123)" << std::endl;
        v1.SetValue(1, -10.123);
        std::cout << "v1.GetValue(1): " << v1.GetValue(1) << std::endl;
        v1.PrintData(0, v1.Length());

        IVector<double>* IVector1Ptr = &v1;
        IVector1Ptr->Transpose();
        std::cout << "typeid(IVector1Ptr).name(): " << typeid(IVector1Ptr).name() << std::endl;
        IVector1Ptr->Print();
        IVector1Ptr->PrintData(0, IVector1Ptr->Length());

        IVector<double>* IVectorSplitResultPtr = IVectorHelper::Split(IVector1Ptr, IVector1Ptr, DataLocation::RAM);
        IVectorSplitResultPtr->Print();
        IVectorSplitResultPtr->PrintData();

        ConsoleHelper::PrintLine("ClearData()");
        IVectorSplitResultPtr->ClearData();
        IVectorSplitResultPtr->Print();
        IVectorSplitResultPtr->PrintData();
    }

    static void VectorRamGpus_ConsoleUI()
    {
        std::cout << "VectorRamGpus_ConsoleUI" << std::endl;
        bool res;

        ConsoleHelper::PrintLine("VectorRamGpus<double> v1;");
        VectorRamGpus<double> v1;
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        ConsoleHelper::PrintLine("----------\n");
        //ConsoleHelper::WaitAnyKey();

        
        unsigned N_RAM = 800000000;
        ConsoleHelper::PrintKeyValue("N_RAM", N_RAM);
        ConsoleHelper::PrintLine("auto res = v1.Add(DataLocation::RAM, N_RAM);");
        res = v1.Add(DataLocation::RAM, N_RAM);
        ConsoleHelper::PrintKeyValue("res", res);
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        ConsoleHelper::PrintLine("----------\n");
        //ConsoleHelper::WaitAnyKey();
        
        
        unsigned N_GPU0 = 800000000;
        ConsoleHelper::PrintKeyValue("N_GPU0", N_GPU0);
        ConsoleHelper::PrintLine("res = v1.Add(DataLocation::GPU0, N_GPU0);");
        res = v1.Add(DataLocation::GPU0, N_GPU0);
        ConsoleHelper::PrintKeyValue("res", res);
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        ConsoleHelper::PrintLine("----------\n");
        //ConsoleHelper::WaitAnyKey();

        
        unsigned N_GPU1 = N_GPU0;
        ConsoleHelper::PrintKeyValue("N_GPU1", N_GPU1);
        ConsoleHelper::PrintLine("res = v1.Add(DataLocation::GPU1, N_GPU1);");
        //res = v1.Add(DataLocation::GPU1, N_GPU1);
        ConsoleHelper::PrintKeyValue("res", res);
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
        ConsoleHelper::PrintLine("----------\n");
        //ConsoleHelper::WaitAnyKey();

        ConsoleHelper::PrintLine("auto size = v1.Size();");
        auto size = v1.Length();
        ConsoleHelper::PrintKeyValue("size", size);
        ConsoleHelper::PrintLine("----------\n");
        ConsoleHelper::WaitAnyKey();

        ConsoleHelper::PrintLine("v1.PrintData(0, size);");
        //v1.PrintData(0, size);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.InitByVal(0.01);");
        v1.InitByVal(0.01);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.PrintData(0, size-1);");
        //v1.PrintData(0, size-1);
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

        ConsoleHelper::PrintLine("bool isValueSetted = v1.SetValue(size-1, 23.456);");
        isValueSetted = v1.SetValue(size-1, 23.456);
        ConsoleHelper::PrintKeyValue("isValueSetted", isValueSetted);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("val = v1.GetValue(size-1);");
        val = v1.GetValue(size-1);
        ConsoleHelper::PrintKeyValue("val", val);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.PrintData(0, size);");
        //v1.PrintData(0, size);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.Multiply(2);");
        auto start1 = std::chrono::high_resolution_clock::now();
        v1.Multiply(2);
        auto stop1 = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(stop1 - start1);        
        long long time_mks = duration.count();
        ConsoleHelper::PrintKeyValue("time_mks", time_mks);
        ConsoleHelper::PrintLine("v1.PrintData(N_RAM - 10, 20);");
        v1.PrintData(N_RAM - 10, 20);
        ConsoleHelper::PrintLine("----------\n");

        ConsoleHelper::PrintLine("v1.Multiply(2, true);");
        auto start2 = std::chrono::high_resolution_clock::now();
        v1.Multiply(2, true);
        auto stop2 = std::chrono::high_resolution_clock::now();
        auto duration2 = duration_cast<std::chrono::microseconds>(stop2 - start2);        
        time_mks = duration2.count();
        ConsoleHelper::PrintKeyValue("time_mks", time_mks);
        ConsoleHelper::PrintLine("v1.PrintData(N_RAM - 10, 20);");
        v1.PrintData(N_RAM - 10, 20);
        ConsoleHelper::PrintLine("----------\n");

        //bool isClear = ConsoleHelper::GetBoolFromUser("Do you want clear vector data? (y/n)");
        bool isClear = true;
        if(isClear)
        {
            ConsoleHelper::PrintLine("v1.Clear();");
            v1.Clear();
        }
        
        ConsoleHelper::PrintLine("v1.Print();");
        v1.Print();
    }

};