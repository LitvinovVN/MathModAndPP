#pragma once

/// @brief Абстрактный класс, моделирующий вектор в N-мерном пространстве
/// @tparam T 
template<typename T>
class IVector
{
public:
    // Тип вектора
    VectorType vectorType = VectorType::VectorRow;
    // Инициализирует все элементы вектора указанным значением
    virtual void InitByVal(T val) = 0;
    // Выводит в консоль сведения об объекте
    virtual void Print() const = 0;
    // Выводит в консоль элементы вектора в заданном диапазоне
    virtual void PrintData(unsigned long long indStart,
        unsigned long long length) const = 0;
    // Возвращает количество элементов вектора
    virtual size_t Size() const = 0;
    // Возвращает значение элемента вектора, расположенного по указанному индексу
    virtual T GetValue(unsigned long long index) const = 0;
    // Устанавливает значение элемента вектора, расположенного по указанному индексу
    virtual bool SetValue(unsigned long long index, T value) = 0;
};
