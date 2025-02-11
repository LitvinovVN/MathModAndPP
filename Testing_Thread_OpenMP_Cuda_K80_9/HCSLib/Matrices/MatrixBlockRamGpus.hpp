#pragma once

#include <iostream>

#include "IMatrix.hpp"

/// @brief Блочная матрица на выч. узле (RAM+GPUs)
class MatrixBlockRamGpus : IMatrix
{
    unsigned mb;
    unsigned nb;
    unsigned n;

public:
    MatrixBlockRamGpus(unsigned mb, unsigned nb, unsigned n)
     : mb(mb), nb(nb), n(n)
    {}

    virtual MatrixType GetMatrixType() const override
    {
        return MatrixType::MatrixBlockRamGpus;
    }

    /// @brief Возвращает объём занятой оперативной памяти в байтах
    /// @return Объём занятой оперативной памяти в байтах
    virtual unsigned long long GetSize() const override
    {
        throw std::runtime_error("Net realized!");
    }

    unsigned long long GetM() const override
    {
        return mb*n;
    }

    unsigned long long GetN() const override
    {
        return nb*n;
    }

    /// @brief Возвращает значение элемента матрицы по указанному индексу
    /// @param i Индекс строки
    /// @param j Индекс столбца
    /// @return Элемент (i, j)
    virtual double GetValue(unsigned long long i, unsigned long long j) const override
    {
        //
        //
        //
        //
        //

        return 0;
    }

    virtual double operator()(unsigned long long i, unsigned long long j) const override
    {
        return GetValue(i, j);
    }

    void Print(PrintParams pp = PrintParams{}) const override
    {
        std::cout << "MatrixBlockRamGpus:";
        std::cout << pp.startMes;
        std::cout << "this"<< pp.splitterKeyValue << this;
        std::cout << pp.splitter;
        std::cout << "M" << pp.splitterKeyValue << GetM();
        std::cout << pp.splitter;
        std::cout << "N" << pp.splitterKeyValue << GetN();
        std::cout << pp.splitter;
        std::cout << "mb" << pp.splitterKeyValue << mb;
        std::cout << pp.splitter;
        std::cout << "nb" << pp.splitterKeyValue << nb;
        std::cout << pp.splitter;
        std::cout << "n" << pp.splitterKeyValue << n;
        std::cout << pp.endMes;

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

    /////////
    
    /// @brief Добавляет единичную матрицу по указанным координатам
    /// @param bi Индекс строки 
    /// @param bj Индекс столбца
    void AddE(unsigned bi, unsigned bj)
    {

    }
    /////////

};