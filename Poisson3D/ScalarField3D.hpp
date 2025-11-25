#pragma once
#include <vector>
#include "Coord3D.hpp"
#include "Grid3DParams.hpp"
#include "XYMask.hpp"

struct ScalarField3D
{
    Coord3D coord;// Координаты точки (0,0,0) в пространстве
    Grid3DParams gridParams;// Параметры сетки
    std::vector<std::vector<std::vector<double>>> data;//x,y,z
    XYMask xyMask;
    // Контейнер масок вдоль Oz (только для частично заполненных линий)
    ZMaskRepository zMaskRepository;

    ScalarField3D(Coord3D coord, Grid3DParams gridParams, XYMask xyMask)
        : gridParams(gridParams), coord(coord), xyMask(xyMask)
    {
        data.resize(gridParams.nx);
        for (size_t i = 0; i < gridParams.nx; i++)
        {
            data[i].resize(gridParams.ny);
            for (size_t j = 0; j < gridParams.ny; j++)
            {
                data[i][j].resize(gridParams.nz);
            }            
        }        
    }    

    void PrintInfo()
    {
        std::cout << "ScalarField3D::PrintInfo(): " << this << "; ";
        std::cout << "Coordinates: "; coord.Print(); std::cout << "; ";
        std::cout << "n (" << gridParams.nx << ", " << gridParams.ny << ", " << gridParams.nz << "); ";
        std::cout << "h (" << gridParams.hx << ", " << gridParams.hy << ", " << gridParams.hz << "); ";
        std::cout << "data sizes (" << data.size()<< "); ";
        std::cout << std::endl;
    }

    void PrintData()
    {
        std::cout << "----- Printing data array START -----" <<  std::endl;
        std::cout << "data sizes: " << data.size() <<  std::endl;

        if(gridParams.nx > 100 || gridParams.ny > 100 || gridParams.nz > 100)
        {
            std::cout << "! So many elements. Printing stopped..." << std::endl;
            return;
        }

        for (size_t i = 0; i < gridParams.nx; i++)
        {
            std::cout << "     i = " << i << std::endl;
            for (size_t j = 0; j < gridParams.ny; j++)
            {
                std::cout << "j = " << j << ": ";
                for (size_t k = 0; k < gridParams.nz; k++)
                {
                    std::cout << data[i][j][k] << " ";
                }
                std::cout << " | j = " << j << "; k = 0..." << gridParams.nz-1 << std::endl;
            }            
            std::cout << "----------------------" <<  std::endl;
        }
        std::cout << "----- Printing data array END -----" <<  std::endl;
    }


};
