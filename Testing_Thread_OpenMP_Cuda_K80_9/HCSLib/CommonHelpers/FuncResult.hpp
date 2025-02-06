#pragma once

#include "PrintParams.hpp"

template<typename T>
struct FuncResult
{
    bool        status{};
    T           result{};
    long long   time{};

    FuncResult()
    { }

    FuncResult(bool status, T result, long long time) : 
        status(status), result(result), time(time)
    { }

    void Print(PrintParams pp = PrintParams{})
    {
        std::cout << pp.startMes;
        std::cout << "status" << pp.splitterKeyValue << std::boolalpha << status;
        std::cout << pp.splitter;
        std::cout << "result" << pp.splitterKeyValue << result;
        std::cout << pp.splitter;
        std::cout << "time" << pp.splitterKeyValue << time << " mks";
        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
    
    static bool compare(const FuncResult<T>& left, const FuncResult<T>& right) 
    { 
        return left.time < right.time; 
    }
};
