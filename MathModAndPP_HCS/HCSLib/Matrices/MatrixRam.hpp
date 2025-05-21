#pragma once

#include "../CommonHelpers/PrintParams.hpp"
#include "IMatrix.hpp"

/// @brief Класс "Матрица в RAM"
class MatrixRam : IMatrix
{
public:
    const unsigned long long M{};
    const unsigned long long N{};

    MatrixRam()
    { }

    MatrixRam(unsigned long long M,
    unsigned long long N) :
    M(M), N(N)
    {
    }

    /// @brief Возвращает объём занятой оперативной памяти в байтах
    /// @return Объём занятой оперативной памяти в байтах
    unsigned long long GetSize() const override
    {
        long long size = sizeof(this);
        return size;
    }

    void Print(PrintParams pp = PrintParams{}) const override
    {
        std::cout << "void MatrixRam::Print() const " << std::endl;
        std::cout << pp.startMes;

        std::cout << "M" << pp.splitterKeyValue << GetM();
        std::cout << pp.splitter;
        std::cout << "N" << pp.splitterKeyValue << GetN();
        std::cout << pp.splitter;
        std::cout << "GetMatrixType()" << pp.splitterKeyValue << GetMatrixType();
        std::cout << pp.splitter;
        std::cout << "GetSize()" << pp.splitterKeyValue << GetSize();

        std::cout << pp.endMes;

        PrintMatrix();

        if(pp.isEndl)
            std::cout << std::endl;
    }

    void PrintMatrix() const override
    {
        PrintMatrix(0, GetM(), 0, GetN());
    }

    void PrintMatrix(unsigned long long ind_row_start,
    unsigned long long num_rows,
    unsigned long long ind_col_start,
    unsigned long long num_cols) const override
    {
        std::cout << "\nPrinting matrix: \n";
        std::cout << "Rows: " << ind_row_start << "..";
        std::cout << (ind_row_start + num_rows - 1) << "; ";
        std::cout << "Cols: " << ind_col_start << "..";
        std::cout << (ind_col_start + num_cols - 1) << "]\n";
        for (auto i = ind_row_start; i < ind_row_start + num_rows; i++)
        {
            for (auto j = ind_col_start; j < ind_col_start + num_cols; j++)
            {
                std::cout << GetValue(i, j) << " ";
            }
            std::cout << "\n";
        }        
    }
};