#pragma once
#include "PhysField3DZ.hpp"

struct PhysField3DAlgGrad
{
    inline static void InitByGlobalIndex(IPhysField3D *physField3D)
    {
        size_t Nx = physField3D->GetNx();
        size_t Ny = physField3D->GetNy();
        size_t Nz = physField3D->GetNz();

#pragma omp parallel for
        // for (size_t k = 0; k < Nz; k++) // Медленно
        for (size_t i = 0; i < Nx; i++)
        {
            // size_t offsetNyNz = i * Nz * Ny;
            for (size_t j = 0; j < Ny; j++)
            {
                // size_t offsetNyNzNz = offsetNyNz + j * Nz;
                for (size_t k = 0; k < Nz; k++) // Быстро
                {
                    double val = k + 2 * j + 3 * i;
                    // double val = k + j * Nz + i * Nz * Ny;
                    // double val = k + offsetNyNzNz;
                    physField3D->SetValue(i, j, k, val);
                    // std::cout << physField3D->GetValue(i, j, k) << " ";
                }
            }
        }
        // std::cout << std::endl;
    }

    static void Print_p0_p7(std::string message,
        double* p0, double* p1, double* p2, double* p3,
        double* p4, double* p5, double* p6, double* p7)
    {
        std::cout << "    " << message << "\n";
        if(p0 != nullptr)
            std::cout << "      p0: " << p0 << "| val: " << *p0 << "\n";
        else
            std::cout << "      p0: null\n";
        if(p1 != nullptr)
            std::cout << "      p1: " << p1 << "| val: " << *p1 << "\n";
        else
            std::cout << "      p1: null\n";
        if(p2 != nullptr)
            std::cout << "      p2: " << p2 << "| val: " << *p2 << "\n";
        else
            std::cout << "      p2: null\n";
        if(p3 != nullptr)
            std::cout << "      p3: " << p3 << "| val: " << *p3 << "\n";
        else
            std::cout << "      p3: null\n";
        if(p4 != nullptr)
            std::cout << "      p4: " << p4 << "| val: " << *p4 << "\n";
        else
            std::cout << "      p4: null\n";
        if(p5 != nullptr)
            std::cout << "      p5: " << p5 << "| val: " << *p5 << "\n";
        else
            std::cout << "      p5: null\n";
        if(p6 != nullptr)
            std::cout << "      p6: " << p6 << "| val: " << *p6 << "\n";
        else
            std::cout << "      p6: null\n";
        if(p7 != nullptr)
            std::cout << "      p7: " << p7 << "| val: " << *p7 << "\n";
        else
            std::cout << "      p7: null\n";
    }

    static void GradX(PhysField3DZ *src, PhysField3DZ *res)
    {
        //std::cout << "--- START PhysField3DAlgGrad::GradX(const PhysField3DZ& src, PhysField3DZ& res) START ---\n";
        // 0. Инициализация данных
        size_t Nbx = src->GetNbx();  // Количество блоков сетки вдоль оси Ox
        size_t Nby = src->GetNby();  // Количество блоков сетки вдоль оси Oy
        size_t Nbz = src->GetNbz();  // Количество блоков сетки вдоль оси Oz
        size_t Nbyz = Nby*Nbz;
        size_t Nnbx = src->GetNnbx();// Количество узлов в блоке вдоль оси Ox
        size_t Nnby = src->GetNnby();// Количество узлов в блоке вдоль оси Oy
        size_t Nnbz = src->GetNnbz();// Количество узлов в блоке вдоль оси Oz
        size_t Nnb = Nnbx*Nnby*Nnbz; // Количество узлов в блоке
        size_t blockOffsetX = Nbyz*Nnb;// Смещение между блоками вдоль оси Ox
        double Hx = src->GetHx();    // Шаг сетки вдоль оси Ox
        double khx  = 1/Hx;          // 1/hx
        double k2hx = 1/(2*Hx);      // 1/(2hx)
        double* data_src = src->GetData();// Указатель на массив данных скалярного поля
        double* data_res = res->GetData();// Указатель на массив данных градиента
        //std::cout << "Nb: {" << Nbx << ", " << Nby << ", " << Nbz << "}; Nbyz: " << Nbyz << "\n";
        //std::cout << "Nnb: {" << Nnbx << ", " << Nnby << ", " << Nnbz << "} (" << Nnb << ")\n";
        //std::cout << "blockOffsetX: " << blockOffsetX << "\n";
        //std::cout << "Hx: " << Hx << "; khx: " << khx << "; k2hx: " << k2hx << "\n";
        //std::cout << "data_src: " << data_src << "; data_res: " << data_res << "\n";

        // Предыдущий блок
        double *p0prev = nullptr;
        double *p1prev = nullptr;
        double *p2prev = nullptr;
        double *p3prev = nullptr;
        double *p4prev = nullptr;
        double *p5prev = nullptr;
        double *p6prev = nullptr;
        double *p7prev = nullptr;
        // Текущий блок
        double *p0 = nullptr;
        double *p1 = nullptr;
        double *p2 = nullptr;
        double *p3 = nullptr;
        double *p4 = nullptr;
        double *p5 = nullptr;
        double *p6 = nullptr;
        double *p7 = nullptr;
        // Следующий блок
        double *p0next = nullptr;
        double *p1next = nullptr;
        double *p2next = nullptr;
        double *p3next = nullptr;
        double *p4next = nullptr;
        double *p5next = nullptr;
        double *p6next = nullptr;
        double *p7next = nullptr;

        // Градиенты по оси Ox
        double p0gradX = 0;
        double p1gradX = 0;
        double p2gradX = 0;
        double p3gradX = 0;
        double p4gradX = 0;
        double p5gradX = 0;
        double p6gradX = 0;
        double p7gradX = 0;

        double *ptr_p0gradX_res = nullptr;
        double *ptr_p1gradX_res = nullptr;
        double *ptr_p2gradX_res = nullptr;
        double *ptr_p3gradX_res = nullptr;
        double *ptr_p4gradX_res = nullptr;
        double *ptr_p5gradX_res = nullptr;
        double *ptr_p6gradX_res = nullptr;
        double *ptr_p7gradX_res = nullptr;

        // 1. Перебираем блоки сетки в плоскости zOy
        #pragma omp parallel for
        for (size_t gb_zOy = 0; gb_zOy < Nbyz; gb_zOy++)
        {
            //std::cout << "gb_zOy: " << gb_zOy << "\n";
            // Перебираем блоки вдоль оси Ox
            for (size_t i = 0; i < Nbx; i++)
            {
                //std::cout << "  i: " << i << "; ";
                // Смещение начала блока сетки (точки 0 (0,0,0)) относительно начала массива
                size_t p0offset = gb_zOy*Nnb + i*blockOffsetX;
                //std::cout << "p0offset: " << p0offset << "\n";
                // Указатель на начало блока сетки - точка 0 (0,0,0)
                p0 = data_src + p0offset;
                // Указатели на точки p1-p7 текущего блока
                p1 = p0 + 1;
                p2 = p0 + 2;
                p3 = p0 + 3;
                p4 = p0 + 4;
                p5 = p0 + 5;
                p6 = p0 + 6;
                p7 = p0 + 7;

                // Указатели на точки p1-p7 предыдущего блока
                if(i == 0)// Для первого блока предыдущего нет
                {
                    p0prev = nullptr;
                    p1prev = nullptr;
                    p2prev = nullptr;
                    p3prev = nullptr;
                    p4prev = nullptr;
                    p5prev = nullptr;
                    p6prev = nullptr;
                    p7prev = nullptr;
                }
                else
                {
                    p0prev = p0 - blockOffsetX;
                    p1prev = p1 - blockOffsetX;
                    p2prev = p2 - blockOffsetX;
                    p3prev = p3 - blockOffsetX;
                    p4prev = p4 - blockOffsetX;
                    p5prev = p5 - blockOffsetX;
                    p6prev = p6 - blockOffsetX;
                    p7prev = p7 - blockOffsetX;
                }

                // Указатели на точки p1-p7 следующего блока
                if(i == Nbx-1)// Для последнего блока следующего нет
                {
                    p0next = nullptr;
                    p1next = nullptr;
                    p2next = nullptr;
                    p3next = nullptr;
                    p4next = nullptr;
                    p5next = nullptr;
                    p6next = nullptr;
                    p7next = nullptr;
                }
                else
                {
                    p0next = p0 + blockOffsetX;
                    p1next = p1 + blockOffsetX;
                    p2next = p2 + blockOffsetX;
                    p3next = p3 + blockOffsetX;
                    p4next = p4 + blockOffsetX;
                    p5next = p5 + blockOffsetX;
                    p6next = p6 + blockOffsetX;
                    p7next = p7 + blockOffsetX;
                }

                //Print_p0_p7("--Prev block", p0prev, p1prev, p2prev, p3prev, p4prev, p5prev, p6prev, p7prev);
                //Print_p0_p7("--Current block", p0, p1, p2, p3, p4, p5, p6, p7);
                //Print_p0_p7("--Next block", p0next, p1next, p2next, p3next, p4next, p5next, p6next, p7next);
            
                if(i == 0)// Вычисляем градиент для первого блока
                {
                    p0gradX = khx  * ( (*p4) - (*p0) );
                    p1gradX = khx  * ( (*p5) - (*p1) );
                    p2gradX = khx  * ( (*p6) - (*p2) );
                    p3gradX = khx  * ( (*p7) - (*p3) );
                    p4gradX = k2hx * ( (*p0next) - (*p0) );
                    p5gradX = k2hx * ( (*p1next) - (*p1) );
                    p6gradX = k2hx * ( (*p2next) - (*p2) );
                    p7gradX = k2hx * ( (*p3next) - (*p3) );                    
                }
                else if(i == Nbx-1)// Вычисляем градиент для последнего блока
                {
                    p0gradX = k2hx  * ( (*p4) - (*p4prev) );
                    p1gradX = k2hx  * ( (*p5) - (*p5prev) );
                    p2gradX = k2hx  * ( (*p6) - (*p6prev) );
                    p3gradX = k2hx  * ( (*p7) - (*p7prev) );
                    p4gradX = khx * ( (*p4) - (*p0) );
                    p5gradX = khx * ( (*p5) - (*p1) );
                    p6gradX = khx * ( (*p6) - (*p2) );
                    p7gradX = khx * ( (*p7) - (*p3) );
                }
                else// Вычисляем градиент для внутренних блоков
                {
                    p0gradX = k2hx  * ( (*p4) - (*p4prev) );
                    p1gradX = k2hx  * ( (*p5) - (*p5prev) );
                    p2gradX = k2hx  * ( (*p6) - (*p6prev) );
                    p3gradX = k2hx  * ( (*p7) - (*p7prev) );
                    p4gradX = k2hx * ( (*p0next) - (*p0) );
                    p5gradX = k2hx * ( (*p1next) - (*p1) );
                    p6gradX = k2hx * ( (*p2next) - (*p2) );
                    p7gradX = k2hx * ( (*p3next) - (*p3) );
                }
                /*std::cout << "p0gradX: " << p0gradX << "\n";
                std::cout << "p1gradX: " << p1gradX << "\n";
                std::cout << "p2gradX: " << p2gradX << "\n";
                std::cout << "p3gradX: " << p3gradX << "\n";
                std::cout << "p4gradX: " << p4gradX << "\n";
                std::cout << "p5gradX: " << p5gradX << "\n";
                std::cout << "p6gradX: " << p6gradX << "\n";
                std::cout << "p7gradX: " << p7gradX << "\n";//*/
            
                // Сохраняем блок результатов
                // Указатель на начало блока сетки - точка 0 (0,0,0)
                ptr_p0gradX_res = data_res + p0offset;
                // Указатели на точки p1-p7 текущего блока
                ptr_p1gradX_res = ptr_p0gradX_res + 1;
                ptr_p2gradX_res = ptr_p0gradX_res + 2;
                ptr_p3gradX_res = ptr_p0gradX_res + 3;
                ptr_p4gradX_res = ptr_p0gradX_res + 4;
                ptr_p5gradX_res = ptr_p0gradX_res + 5;
                ptr_p6gradX_res = ptr_p0gradX_res + 6;
                ptr_p7gradX_res = ptr_p0gradX_res + 7;

                // Записываем блок
                *ptr_p0gradX_res = p0gradX;
                *ptr_p1gradX_res = p1gradX;
                *ptr_p2gradX_res = p2gradX;
                *ptr_p3gradX_res = p3gradX;
                *ptr_p4gradX_res = p4gradX;
                *ptr_p5gradX_res = p5gradX;
                *ptr_p6gradX_res = p6gradX;
                *ptr_p7gradX_res = p7gradX;
            }
        }

        //std::cout << "--- END PhysField3DAlgGrad::GradX(const PhysField3DZ& src, PhysField3DZ& res) END ---\n\n";
    }
};
