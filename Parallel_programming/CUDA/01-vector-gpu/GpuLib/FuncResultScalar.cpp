#pragma once

#include <iostream>
#include <vector>

/// @brief Структура для хранения результата запуска 
template<typename T = double>
struct FuncResultScalar
{
    // Статус выполнения функции
    bool Status = true;
    // Результат функции
    T Result;
    // Время выполнения функции, мкс
    long long Time_mks;

    void Print()
    {
        std::cout << "["   << Status 
                  << ", "  << Result
                  << ", "  << Time_mks
                  << " mks]"   << std::endl;
    }    
};

std::ostream& operator<<(std::ostream &stream, const FuncResultScalar<> &funcResultScalar)
{
    return stream   << "["       << funcResultScalar.Status 
                    << ", "      << funcResultScalar.Result
                    << ", "      << funcResultScalar.Time_mks
                    << " mks]";
}

bool compare(const FuncResultScalar<double>& left, const FuncResultScalar<double>& right) 
{ 
    return left.Time_mks < right.Time_mks; 
}

void print(std::string message, std::vector<FuncResultScalar<double>> results)
{
    std::cout << message << std::endl;
    for(auto& item : results)
    {
        item.Print();
        //std::cout << std::endl;
    }
}