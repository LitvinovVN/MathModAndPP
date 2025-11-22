#pragma once
#include <iostream>
#include "IPhysField3D.hpp"

class PhysField3DZ : public IPhysField3D
{
    std::string Description;
    size_t Nbx, Nby, Nbz;// Количество блоков сетки    
    size_t Nnbx, Nnby, Nnbz;// Количество узлов в блоке
    size_t Nx, Ny, Nz;
    double Hx, Hy, Hz; 
    double* data = nullptr;

    inline size_t GetIndex(size_t i, size_t j, size_t k) const
    {
        size_t iloc = i%Nnbx;
        size_t jloc = j%Nnby;
        size_t kloc = k%Nnbz;
        size_t ind = kloc + jloc*Nnbz + iloc*Nnbz*Nnby;
        size_t ib = i/Nnbx;
        size_t jb = j/Nnby;
        size_t kb = k/Nnbz;
        size_t gbInd = kb+jb*Nbz+ib*Nby*Nbz;
        size_t gInd = gbInd*(Nnbx*Nnby*Nnbz)+ind;
        return gInd;
        //return k + j*Nz + i*Ny*Nz;// ----
    }

    inline size_t GetIndexX(size_t globalInd) const
    {
        size_t gbInd = globalInd / (Nnbx*Nnby*Nnbz);
        size_t ib = gbInd / (Nbz*Nby);
        size_t ind = globalInd % (Nnbx*Nnby*Nnbz);
        size_t iloc = ind / (Nnby*Nnbz);
        size_t i = ib*Nnbx + iloc;
        //std::cout << " [" << gbInd << " " << ib << " " << iloc << " " << i << "] ";
        return i;
        //return globalInd / (Nz*Ny);// ----
    }

    inline size_t GetIndexY(size_t globalInd) const
    {
        size_t gbInd = globalInd / (Nnbx*Nnby*Nnbz);
        size_t jb = gbInd / Nbz % Nby;
        size_t ind = globalInd % (Nnbx*Nnby*Nnbz);
        size_t jloc = ind / Nnbz % Nnby;
        size_t j = jb*Nnby + jloc;
        //std::cout << " [" << gbInd << " " << jb << " " << ind << " " << jloc << " " << j << "] ";
        return j;
        //return (globalInd % (Nz*Ny))/Nz;// ----
    }

    inline size_t GetIndexZ(size_t globalInd) const
    {
        size_t gbInd = globalInd / (Nnbx*Nnby*Nnbz);
        size_t kb = gbInd % Nbz;
        size_t ind = globalInd % (Nnbx*Nnby*Nnbz);
        size_t kloc = ind % Nnbz;
        size_t k = kb*Nnbz + kloc;
        //std::cout << " [" << gbInd << " " << kb << " " << ind << " " << kloc << " " << k << "] ";
        return k;
        //return globalInd % Nz;// ----
    }

public:
    PhysField3DZ(size_t Nbx, size_t Nby, size_t Nbz,
        size_t Nnbx, size_t Nnby, size_t Nnbz,
        double Hx, double Hy, double Hz,
        std::string Description = "")
            : Nbx(Nbx), Nby(Nby), Nbz(Nbz),
              Nnbx(Nnbx), Nnby(Nnby), Nnbz(Nnbz),
              Nx(Nbx*Nnbx), Ny(Nby*Nnby), Nz(Nbz*Nnbz), 
              Hx(Hx), Hy(Hy), Hz(Hz), Description(Description)
    {
        data = new double[Nx*Ny*Nz];
    }
    ~PhysField3DZ() override
    {
        std::cout << "~PhysField3DZ()...";
        if(data == nullptr) return;

        delete[] data;
        std::cout << "OK\n";
    }

    std::string GetDescription() const override
    {
        return Description;
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

    inline double GetHx() const override
    {
        return Hx;
    }

    inline double GetHy() const override
    {
        return Hy;
    }

    inline double GetHz() const override
    {
        return Hz;
    }
    
    inline double GetValue(size_t i, size_t j, size_t k) const override
    {
        return data[GetIndex(i,j,k)];
    }

    inline void SetValue(size_t i, size_t j, size_t k, double value) override
    {
        data[GetIndex(i,j,k)] = value;
    }

    void Print() const override
    {
        std::cout << "Description: " << GetDescription() << "\n";
        std::cout << " - object type: PhysField3DZ\n";
        std::cout << " - object address: " << this << "\n";
        std::cout << " - data address: " << data << "\n";
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
