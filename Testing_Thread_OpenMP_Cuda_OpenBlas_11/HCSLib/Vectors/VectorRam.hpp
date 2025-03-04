#pragma once

template<typename T>
class VectorRam : public IVector<T>
{
public:
    T* data;
    size_t size;

    VectorRam(size_t size) : size(size)
    {
        data = new T[size];
    }

    ~VectorRam()
    {
        delete[] data;
    }

    void InitByVal(T val) override
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] = val;
        }        
    }

    void Print() const override
    {
        for (size_t i = 0; i < size; i++)
        {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;     
    }

    /// @brief Выводит в консоль элементы вектора в заданном диапазоне
    void PrintData(unsigned long long indStart,
        unsigned long long length) const override
    {
        throw std::runtime_error("Not realized!");
    }

    size_t Size() const override
    {
        return size;
    }

    /// @brief Возвращает значение элемента вектора, расположенного по указанному индексу
    T GetValue(unsigned long long index) const override
    {
        throw std::runtime_error("Not realized!");
    }

    /// @brief Устанавливает значение элемента вектора, расположенного по указанному индексу
    bool SetValue(unsigned long long index, T value) override
    {        
        throw std::runtime_error("Not realized!");
    }

};