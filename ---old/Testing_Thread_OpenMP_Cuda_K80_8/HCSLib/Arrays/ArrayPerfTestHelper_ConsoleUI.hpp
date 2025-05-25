#pragma once

#include <iostream>

struct ArrayPerfTestHelper_ConsoleUI
{

static void SumOpenMP_ConsoleUI()
{
    std::cout << "ArrayPerfTestHelper_ConsoleUI::SumOpenMP_ConsoleUI()\n";
    
    unsigned iterNum = ConsoleHelper::GetUnsignedIntFromUser("Enter iterations number: ");

    unsigned long long arrayLengthMin = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length min: ");
    unsigned long long arrayLengthMax = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length max: ");
    unsigned long long arrayLengthStep = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length step: ");

    unsigned cpuThreadNumMin = ConsoleHelper::GetUnsignedIntFromUser("Enter num cpu threads min: ");
    unsigned cpuThreadNumMax = ConsoleHelper::GetUnsignedIntFromUser("Enter num cpu threads max: ");
    unsigned cpuThreadNumStep = ConsoleHelper::GetUnsignedIntFromUser("Enter num cpu threads step: ");

    
    /*std::cout 
        << arrayLengthMin << " "
        << arrayLengthMax << " "
        << arrayLengthStep << " "
        << cpuThreadNumMin << " "
        << cpuThreadNumMax << " "
        << cpuThreadNumStep << " "
        << iterNum << " "
        << std::endl;*/
    
    DataTypes dataTypes;
    dataTypes.Add(DataTypeEnum::dt_float);
    dataTypes.Add(DataTypeEnum::dt_double);
    PerfTestParamsData perfTestParamsData(dataTypes, arrayLengthMin, arrayLengthMax, arrayLengthStep);
    PerfTestParamsCpu perfTestParamsCpu(cpuThreadNumMin, cpuThreadNumMax, cpuThreadNumStep);
    PerfTestParams perfTestParams(iterNum, perfTestParamsData, perfTestParamsCpu);
    perfTestParams.Print();

    PerfTestResults results = ArrayPerfTestHelper::PerfTest_SumOpenMP(perfTestParams);
    results.Print();
}


static void SumCublas_ConsoleUI()
{
    std::cout << "ArrayPerfTestHelper_ConsoleUI::SumCublas_ConsoleUI()\n";
    
    unsigned iterNum = ConsoleHelper::GetUnsignedIntFromUser("Enter iterations number: ");

    unsigned long long arrayLengthMin = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length min: ");
    unsigned long long arrayLengthMax = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length max: ");
    unsigned long long arrayLengthStep = ConsoleHelper::GetUnsignedLongLongFromUser("Enter array length step: ");

    /*unsigned cpuThreadNumMin = ConsoleHelper::GetUnsignedIntFromUser("Enter num cpu threads min: ");
    unsigned cpuThreadNumMax = ConsoleHelper::GetUnsignedIntFromUser("Enter num cpu threads max: ");
    unsigned cpuThreadNumStep = ConsoleHelper::GetUnsignedIntFromUser("Enter num cpu threads step: ");*/

    
    /*std::cout 
        << arrayLengthMin << " "
        << arrayLengthMax << " "
        << arrayLengthStep << " "
        << cpuThreadNumMin << " "
        << cpuThreadNumMax << " "
        << cpuThreadNumStep << " "
        << iterNum << " "
        << std::endl;*/
    
    DataTypes dataTypes;
    dataTypes.Add(DataTypeEnum::dt_float);
    dataTypes.Add(DataTypeEnum::dt_double);
    PerfTestParamsData perfTestParamsData(dataTypes, arrayLengthMin, arrayLengthMax, arrayLengthStep);
    //PerfTestParamsGpu perfTestParamsGpu(cpuThreadNumMin, cpuThreadNumMax, cpuThreadNumStep);
    //PerfTestParams perfTestParams(iterNum, perfTestParamsData, perfTestParamsCpu);
    PerfTestParams perfTestParams(iterNum, perfTestParamsData);
    perfTestParams.Print();

    PerfTestResults results = ArrayPerfTestHelper::PerfTest_SumCublas(perfTestParams);
    results.Print();
}


};



