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
        FunctionArgument arg_void{};
        FunctionArgument arg_int{1};
        FunctionArgument arg_float{2.f};
        FunctionArgument arg_double{2000.123};
        FunctionArgument arg_ull{12345678909ull};

        float* arr_float {new float[10]{1.1, 2.2, 3.3, 4, 5, 6, 7, 8, 9, 10}};
        FunctionArgument arg_ptr_float{arr_float};

        double* arr_double = new double[10]{0.1, 0.2, 0.3, 0.4, 5, 6, 7, 8, 9, 10};
        FunctionArgument arg_ptr_double{arr_double};

        FunctionArguments func_args;
        func_args.Add(arg_ptr_float);
        func_args.Add(FunctionArgument{0ull});
        func_args.Add(FunctionArgument{9ull});
        func_args.Print(PrintParams{});

        AlgorithmImplementationExecParams execParams;
        execParams.functionArguments = func_args;

        AlgTestingResult res = algorithmImplementationExecutor.Exec(1, execParams);
        res.Print();
    }

};