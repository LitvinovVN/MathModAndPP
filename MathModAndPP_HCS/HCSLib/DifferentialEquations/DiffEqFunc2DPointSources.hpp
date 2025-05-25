#pragma once

#include <vector>

/// @brief Функция правой части дифф. уравнения (точечные источники)
class DiffEqFunc2DPointSources : public IDiffEqFunction
{
    // Массив данных в формате
    // x1 y1 f1 ...
    std::vector<double> data;
public:

    /// @brief Возвращает размерность объекта функции
    Dimension GetDimension() const override
    {
        return Dimension::D2;
    }

    void AddPointSource(double x, double y, double f)
    {
        data.push_back(x);
        data.push_back(y);
        data.push_back(f);
    }

    unsigned int GetNumPointSources() const
    {
        return data.size() / 3;
    }

    double GetValue(double x, double y, double eps = 0.00000001) const
    {
        for(auto i{0ull}; i < data.size(); i += 3)
        {
            auto _x = data[i];
            auto _y = data[i + 1];
            auto _f = data[i + 2];

            if(std::abs(x - _x) < eps && std::abs(y - _y) < eps)
             return _f;
        }

        return 0.0;
    }

    /// @brief Возвращает значение функции в точке
    double GetValue(std::vector<double> coordinates) const override
    {
        if(coordinates.size()<2)
            return 0;
        return GetValue(coordinates[0], coordinates[1]);
    }

    void Print() const override
    {
        for(auto i{0ull}; i < data.size(); i += 3)
        {
            std::cout << data[i] << " "
                      << data[i + 1] << " " 
                      << data[i + 2] << std::endl;
        }
    }
};