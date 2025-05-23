#pragma once

#include "../CommonHelpers/PrintParams.hpp"
#include "MatrixType.hpp"

/// @brief Интерфейс "Матрица"
class IMatrix
{
public:
    /// @brief Возвращает тип матрицы
    /// @return MatrixType
    virtual MatrixType GetMatrixType() const = 0;

    /// @brief Возвращает объём занятой оперативной памяти в байтах
    /// @return Объём занятой оперативной памяти в байтах
    virtual unsigned long long GetSize() const = 0;

    /// @brief Возвращает количество строк M
    /// @return M - количество строк
    virtual unsigned long long GetM() const = 0;

    /// @brief Возвращает количество столбцов N
    /// @return N - количество строк
    virtual unsigned long long GetN() const = 0;

    /// @brief Возвращает значение элемента матрицы по указанному индексу
    /// @param i Индекс строки
    /// @param j Индекс столбца
    /// @return Элемент (i, j)
    virtual double GetValue(unsigned long long i, unsigned long long j) const = 0;

    /// @brief Возвращает значение элемента матрицы по указанному индексу
    /// @param i Индекс строки
    /// @param j Индекс столбца
    /// @return Элемент (i, j)
    virtual double operator()(unsigned long long i, unsigned long long j) const = 0;


    /// @brief Выводит в консоль сведения о матрице
    virtual void Print(PrintParams pp = PrintParams{}) const = 0;

    /// @brief Выводит в консоль значения элементов матрицы
    virtual void PrintMatrix() const = 0;

    /// @brief Выводит в консоль значения элементов матрицы в указанном диапазоне
    /// @param ind_row_start Индекс стартовой строки
    /// @param num_rows Количество строк
    /// @param ind_col_start Индекс стартового столбца
    /// @param num_cols Количество столбцов
    virtual void PrintMatrix(unsigned long long ind_row_start,
        unsigned long long num_rows,
        unsigned long long ind_col_start,
        unsigned long long num_cols) const = 0;

};