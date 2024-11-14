#pragma once

#include <iostream>
#include <vector>

/// @brief Структура для хранения результата запуска 
template<typename T>
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

template<typename T>
std::ostream& operator<<(std::ostream &stream, const FuncResultScalar<T> &funcResultScalar)
{
    return stream   << "["       << funcResultScalar.Status 
                    << ", "      << funcResultScalar.Result
                    << ", "      << funcResultScalar.Time_mks
                    << " mks]";
}

template<typename T>
bool compare(const FuncResultScalar<T>& left, const FuncResultScalar<T>& right) 
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