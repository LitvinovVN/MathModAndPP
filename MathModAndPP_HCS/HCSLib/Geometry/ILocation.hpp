#pragma once

/// @brief Интерфейс для объектов, описывающих расположение в пространстве
class ILocation
{
public:
    virtual void Print() const = 0;
};