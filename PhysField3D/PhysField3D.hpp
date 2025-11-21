#pragma once
#include <iostream>
#include "IPhysField3D.hpp"

class PhysField3D : IPhysField3D
{
    size_t Nx, Ny, Nz;
    double* data = nullptr;

    inline size_t GetIndex(size_t i, size_t j, size_t k) const
    {
        return k + j*Nz + i*Ny*Nz;
    }

    inline size_t GetIndexX(size_t globalInd) const
    {
        return globalInd / (Nz*Ny);
    }

    inline size_t GetIndexY(size_t globalInd) const
    {
        return (globalInd % (Nz*Ny))/Nz;
    }

    inline size_t GetIndexZ(size_t globalInd) const
    {
        return globalInd % Nz;
    }

public:
    PhysField3D(size_t Nx, size_t Ny, size_t Nz)
        : Nx(Nx), Ny(Ny), Nz(Nz)
    {
        data = new double[Nx*Ny*Nz];
    }
    ~PhysField3D() override
    {
        if(data == nullptr) return;

        delete[] data;
    }

    inline size_t GetNx() const override
    {
        return Nx;
    }

    inline size_t GetNy() const override
    {
        return Ny;
    }

    inline size_t GetNz() const override
    {
        return Nz;
    }
    
    inline void SetValue(size_t i, size_t j, size_t k, double value) override
    {
        data[GetIndex(i,j,k)] = value;
    }

    inline double GetValue(size_t i, size_t j, size_t k) const override
    {
        return data[GetIndex(i,j,k)];
    }

    void PrintDataArray(size_t indStart, size_t indEnd) const override
    {
        for (size_t i = indStart; i < indEnd; i++)
        {
            std::cout << "ind: " << i;
            std::cout << " (" << GetIndexX(i) << ", " << GetIndexY(i) << ", " << GetIndexZ(i) << "): " << data[i] << std::endl;
        }        
    }

    void PrintDataArray() const override
    {
        PrintDataArray(0, Nx*Ny*Nz);
    }

    void PrintValue(size_t i, size_t j, size_t k) const override 
    {
        std::cout << "(" << i << ", " << j << ", " << k << ") : ";
        size_t ind = GetIndex(i, j, k);
        std::cout << "[" << ind << "] : ";
        std::cout << "val = " << data[ind] << "\n";
    }
};
