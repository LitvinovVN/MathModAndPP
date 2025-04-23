#pragma once

/// @brief Абстрактный класс, моделирующий скалярное значение в разных видах памяти
/// @tparam T 
template<typename T>
class IScalar
{
public:    
    /// @brief Место расположения данных
    DataLocation dataLocation = DataLocation::RAM;
    
    // /// @brief Выводит в консоль сведения об объекте и его значение
    virtual void Print() const = 0;    
    /// @brief Возвращает значение скаляра
    virtual T GetValue() const = 0;
    /// @brief Устанавливает значение скаляра
    virtual bool SetValue(T value) = 0;
    /// @brief Очищает массивы данных
    virtual void ClearData() = 0;
};
