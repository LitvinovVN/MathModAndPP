#pragma once

class IGeometryLocation
{
public:
    IGeometry* geometry;
    //ILocation* location;
    virtual ILocation* GetLocation() = 0;

    /// @brief Вывод сведений об объекте в консоль
    void Print()
    {
        std::cout << "IGeometryLocation address: " << this << std::endl;
        std::cout << "IGeometry address: " << geometry << std::endl;
        geometry->Print();

        ILocation* location = GetLocation();
        std::cout << "ILocation address: " << location << std::endl;
        location->Print();

    }
};