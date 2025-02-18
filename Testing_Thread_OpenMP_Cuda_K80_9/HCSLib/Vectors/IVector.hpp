#pragma once

/// @brief Абстрактный класс, моделирующий вектор в N-мерном пространстве
/// @tparam T 
template<typename T>
class IVector
{
public:
    virtual void InitByVal(T val) = 0;
    virtual void Print() const = 0;
    virtual size_t Size() const = 0;
};
