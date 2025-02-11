#pragma once

#include "MatrixRam.hpp"

/// @brief Единичная матрица
class MatrixRamE : public MatrixRam
{
public:
    MatrixRamE(unsigned long long M, unsigned long long N)
        : MatrixRam(M, N)
    {
    }

    virtual MatrixType GetMatrixType() const override
    {
        return MatrixType::E;
    }

    unsigned long long GetM() const override
    {
        return M;
    }

    unsigned long long GetN() const override
    {
        return N;
    }

    /// @brief Возвращает значение элемента матрицы по указанному индексу
    /// @param i Индекс строки
    /// @param j Индекс столбца
    /// @return Элемент (i, j)
    virtual double GetValue(unsigned long long i, unsigned long long j) const override
    {
        if(i >= M || j >= N)
        {
            std::cout << "!!!!! i: " << i << "; j: " << j << std::endl;
            throw std::runtime_error("ZeroMatrix::GetValue() error!");
        }

        if(i==j)
            return 1;

        return 0;
    }

    virtual double operator()(unsigned long long i, unsigned long long j) const override
    {
        if(i >= M || j >= N)
        {
            std::cout << "!!!!! i: " << i << "; j: " << j << std::endl;
            throw std::runtime_error("ZeroMatrix::operator() error!");
        }

        if(i==j)
            return 1;

        return 0;
    }
};