#pragma once

#include <iostream>

struct ArrayPerfTestHelper_ConsoleUI
{

static void SumOpenMP_ConsoleUI()
{
    std::cout << "ArrayPerfTestHelper_ConsoleUI::SumOpenMP_ConsoleUI()\n";
    
    unsigned long long arrayLengthMin = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length min: ");
    unsigned long long arrayLengthMax = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length max: ");
    unsigned long long arrayLengthStep = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length step: ");

    unsigned cpuThreadNumMin = ConsoleHelper::GetUnsignedIntFromUser("Enter num cpu threads min: ");
    unsigned cpuThreadNumMax = ConsoleHelper::GetUnsignedIntFromUser("Enter num cpu threads max: ");
    unsigned cpuThreadNumStep = ConsoleHelper::GetUnsignedIntFromUser("Enter num cpu threads step: ");

    unsigned iterNum = ConsoleHelper::GetUnsignedIntFromUser("Enter iterations number: ");

    std::string dataTypeString{"double"};

    std::cout 
        << arrayLengthMin << " "
        << arrayLengthMax << " "
        << arrayLengthStep << " "
        << cpuThreadNumMin << " "
        << cpuThreadNumMax << " "
        << cpuThreadNumStep << " "
        << iterNum << " "
        << std::endl;
    
    //PerfTestParamsData perfTestParamsData(arrayLengthMin, arrayLengthMax, arrayLengthStep);
    //PerfTestParamsCpu perfTestParamsCpu(cpuThreadNumMin, cpuThreadNumMax, cpuThreadNumStep);
    //PerfTestParams perfTestParams(perfTestParamsCpu, perfTestParamsData);

    //PerfTestResults<double> results = ArrayPerfTestHelper::SumOpenMP<double>(perfTestParams);
    //results.Print();
}

};



