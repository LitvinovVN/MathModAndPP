#pragma once

#include "MatrixRam.hpp"

/// @brief Нулевая матрица
class MatrixRamZero : public MatrixRam
{
public:
    MatrixRamZero(unsigned long long M, unsigned long long N)
        : MatrixRam(M, N)
    {     
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
        return 0;
    }

    virtual double operator()(unsigned long long i, unsigned long long j) const override
    {
        if(i >= M || j >= N)
        {
            std::cout << "!!!!! i: " << i << "; j: " << j << std::endl;
            throw std::runtime_error("ZeroMatrix::operator() error!");
        }
        return 0;
    }

    virtual MatrixType GetMatrixType() const override
    {
        return MatrixType::Zero;
    }
};