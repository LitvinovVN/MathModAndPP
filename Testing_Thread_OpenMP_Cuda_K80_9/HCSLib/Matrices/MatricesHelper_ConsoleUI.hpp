#pragma once

#include "MatricesHelper.hpp"

/// @brief Вспомогательный класс для работы с матрицами
class MatricesHelper_ConsoleUI
{
public:
    /// @brief Тестирование класса MatrixRamZero
    static void MatrixRamZeroTesting()
    {
        std::cout << "--- void MatrixRamZeroTesting() ---" << std::endl;
        MatrixRamZero z{4, 6};
        z.Print();

        MatrixRamZero* z_ptr = new MatrixRamZero{3, 5};
        z_ptr->Print();

        MatrixRam* mrz_ptr = new MatrixRamZero{2, 4};
        mrz_ptr->Print();

        IMatrix* iz_ptr = (IMatrix*)z_ptr;
        iz_ptr->Print();
    }

    /// @brief Тестирование класса MatrixRamE
    static void MatrixRamETesting()
    {
        std::cout << "--- void MatrixRamETesting() ---" << std::endl;
        MatrixRamE z{4, 6};
        z.Print();
        std::cout << "z(0,0) = " << z(0,0) <<std::endl;
        std::cout << "z(0,1) = " << z(0,1) <<std::endl;
        std::cout << "z(1,0) = " << z(1,0) <<std::endl;
        std::cout << "z(1,1) = " << z(1,1) <<std::endl;

        MatrixRamE* z_ptr = new MatrixRamE{3, 5};
        z_ptr->Print();

        MatrixRam* mrz_ptr = new MatrixRamE{2, 4};
        mrz_ptr->Print();

        IMatrix* iz_ptr = (IMatrix*)z_ptr;
        iz_ptr->Print();

    }

    /// @brief Класс MatrixBlockRamGpus -  блочная матрица в RAM+GPUs
    static void MatrixBlockRamGpus_ConsoleUI()
    {
        std::cout << "MatricesHelper_ConsoleUI::MatrixBlockRamGpus_ConsoleUI()" << std::endl;

        try
        {
            unsigned mb = 4; //ConsoleHelper::GetUnsignedIntFromUser("Enter number of blocks by row, mb: ");
            unsigned nb = 4; //ConsoleHelper::GetUnsignedIntFromUser("Enter number of blocks by column, mb: ");
            unsigned block_size = 5;

            // Создаём нулевую блочную матрицу нужной размерности
            MatrixBlockRamGpus matrix(mb, nb, block_size);
            matrix.Print();
            matrix.PrintMatrix();

            matrix.AddE(1,2);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        

    }
};

