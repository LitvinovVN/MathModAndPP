#pragma once

template<typename T>
struct FuncResult
{
    bool        _status{};
    T           _result{};
    long long   _time{};

    FuncResult()
    { }

    FuncResult(bool status, T result, long long time) : 
        _status(status), _result(result), _time(time)
    { }

    void Print()
    {
        std::cout << "[status: " << std::boolalpha << _status
                  << "; val: " << _result
                  << "; time: " << _time << " mks]" << std::endl;
    }
    
    static bool compare(const FuncResult<T>& left, const FuncResult<T>& right) 
    { 
        return left._time < right._time; 
    }
};
