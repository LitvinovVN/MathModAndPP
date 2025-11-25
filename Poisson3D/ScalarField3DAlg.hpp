#pragma once
#include <iostream>
#include <functional> 
#include "ScalarField3D.hpp"

struct ScalarField3DAlg
{
    /// @brief Инициализация поля функцией
    /// @param scalarField3D 
    /// @param func 
    static void Init(ScalarField3D& scalarField3D, std::function<double(double, double, double)> func)
    {
        std::cout << "----- ScalarField3DAlg::ScalarField3DInit() START -----" << std::endl;
        scalarField3D.PrintInfo();

        std::cout << "scalarField3D.nx: " << scalarField3D.gridParams.nx << std::endl;
        for (unsigned i = 0; i < scalarField3D.gridParams.nx; i++)
        {
            for (unsigned j = 0; j < scalarField3D.gridParams.ny; j++)
            {
                // Если точка не принадлежит маске, то пропускаем ее
                if (scalarField3D.xyMask.IsPoinOutside(i, j))
                    continue;

                for (unsigned k = 0; k < scalarField3D.gridParams.nz; k++)
                {
                    double x = scalarField3D.coord.x + i*scalarField3D.gridParams.hx;
                    double y = scalarField3D.coord.y + j*scalarField3D.gridParams.hy;
                    double z = scalarField3D.coord.z + k*scalarField3D.gridParams.hz;
                    double val = func(x, y, z);
                    //std::cout << val << " ";
                    scalarField3D.data[i][j][k] = val;
                }                
            }            
        }
        
        std::cout << "----- PhysField3DAlg::PhysField3DInit() END -----" << std::endl;
    }
};