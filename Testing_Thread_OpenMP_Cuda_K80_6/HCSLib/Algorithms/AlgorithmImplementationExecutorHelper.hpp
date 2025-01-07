#pragma once

#include <iostream>

struct AlgorithmImplementationExecutorHelper
{
    static void Exec(AlgorithmImplementationExecutor& algorithmImplementationExecutor)
    {
        std::cout << "AlgorithmImplementationExecutorHelper::Exec()\n";

        algorithmImplementationExecutor.SetComputingSystemId(1);
        std::cout << "IsConfigured(): " << algorithmImplementationExecutor.IsConfigured();
        std::cout << std::endl;
        //FunctionArgument arg1;
        //AlgorithmImplementationExecParams execParams;
        //AlgTestingResult res = algorithmImplementationExecutor.Exec(1, AlgorithmImplementationExecParams{});
    }

};