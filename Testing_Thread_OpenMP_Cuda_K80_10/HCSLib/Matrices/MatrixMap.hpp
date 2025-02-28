#pragma once

#include <iostream>
#include <vector>

#include "MatrixMapElement.hpp"

/// @brief Карта матрицы
class MatrixMap
{
    std::vector<std::vector<MatrixMapElement>> mapElements;
public:

    void Print() const
    {
        for (size_t ib = 0; ib < mapElements.size(); ib++)
        {
            std::cout << ib << ": ";
            for (size_t jb = 0; jb < mapElements[ib].size(); jb++)
            {
                std::cout << "[" 
                    << mapElements[ib][jb].columnIndex
                    << "("
                    << mapElements[ib][jb].matrixType
                    << ")] ";
            }
            std::cout << std::endl;
        }
        
    }

    /// @brief Вывод в консоль карты блочной матрицы
    void Print(unsigned mb, unsigned nb) const
    {
        std::cout << "MatrixMap" << std::endl;
        for (size_t ib = 0; ib < mb; ib++)
        {
            for (size_t jb = 0; jb < nb; jb++)
            {
                MatrixType mtype = GetMatrixType(ib, jb);
                switch (mtype)
                {
                case MatrixType::Zero:
                    std::cout << "Z ";
                    break;
                case MatrixType::E:
                    std::cout << "E ";
                    break;
                
                default:
                    break;
                }
                
            }
            std::cout << std::endl;
        }
        
    }

    /// @brief Возвращает тип матрицы, расположенной в блочной матрице по указанному индексу
    /// @param ib 
    /// @param jb 
    /// @return 
    MatrixType GetMatrixType(unsigned ib, unsigned jb) const
    {
        // Проверка
        if(ib >= mapElements.size())
        {
            std::cout << "ib: " << ib;
            std::cout << "\nmapElements.size(): " << mapElements.size();
            throw std::runtime_error("Error in mapElements size!");
        }
            
        // Выбираем строку блочной матрицы
        auto& blockMatrixRow = mapElements[ib];
        // Перебираем блоки в текущей строке
        for (size_t j = 0; j < blockMatrixRow.size(); j++)
        {
            auto curBlock = blockMatrixRow[j];
            if(jb < curBlock.columnIndex)
                return MatrixType::Zero;
            if(jb == curBlock.columnIndex)
                return curBlock.matrixType;
        }
        return MatrixType::Zero;
    }


    void SetRowsNumber(unsigned mb)
    {
        mapElements.clear();
        for (size_t i = 0; i < mb; i++)
        {
            mapElements.push_back(std::vector<MatrixMapElement>{});
        }
        
    }

    /// @brief Добавляет единичную матрицу по указанным координатам
    /// @param bi Индекс строки 
    /// @param bj Индекс столбца
    void AddE(unsigned ib, unsigned jb)
    {
        auto insertingElement = MatrixMapElement{jb, MatrixType::E};

        // Выбираем строку блочной матрицы
        auto& blockMatrixRow = mapElements[ib];                          

        // Перебираем блоки в текущей строке
        for (size_t j = 0; j < blockMatrixRow.size(); j++)
        {
            auto curBlock = blockMatrixRow[j];
            //curCol = curBlock.columnIndex;
            if(jb < curBlock.columnIndex)
            {                
                blockMatrixRow.insert(blockMatrixRow.begin()+j,insertingElement);
                return;
            }
                
        }

        blockMatrixRow.push_back(insertingElement);
    }

};
