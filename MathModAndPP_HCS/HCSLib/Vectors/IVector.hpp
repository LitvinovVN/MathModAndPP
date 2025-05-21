#pragma once

/// @brief Абстрактный класс, моделирующий вектор в N-мерном пространстве
/// @tparam T 
template<typename T>
class IVector
{
public:
    /// @brief Тип вектора
    VectorType vectorType = VectorType::VectorRow;

    /// @brief Место расположения данных
    DataLocation dataLocation = DataLocation::RAM;

    /// @brief Транспонирует вектор (вектор-столбец превращает в вектор-строку и наоборот)
    void Transpose()
    {
        vectorType = (VectorType)!(bool)vectorType;
    }

    /// @brief Инициализирует все элементы вектора указанным значением
    virtual void InitByVal(T val) = 0;
    // /// @brief Выводит в консоль сведения об объекте
    virtual void Print() const = 0;
    /// @brief Выводит в консоль элементы вектора в заданном диапазоне
    virtual void PrintData(unsigned long long indStart,
        unsigned long long length) const = 0;
    /// @brief Выводит в консоль все элементы вектора
    virtual void PrintData() const = 0;
    /// @brief Возвращает длину вектора (количество элементов)
    virtual size_t Length() const = 0;
    /// @brief Возвращает значение элемента вектора, расположенного по указанному индексу
    virtual T GetValue(unsigned long long index) const = 0;
    /// @brief Устанавливает значение элемента вектора, расположенного по указанному индексу
    virtual bool SetValue(unsigned long long index, T value) = 0;
    /// @brief Очищает массивы данных и устанавливает размер вектора в 0
    virtual void ClearData() = 0;
};
