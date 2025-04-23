#pragma once

template<typename T>
class VectorRam : public IVector<T>
{
public:
    T* data;
    size_t length;

    VectorRam(size_t length) : length(length)
    {
        data = new T[length];
    }

    ~VectorRam()
    {
        if(data)
            delete[] data;
    }

    void InitByVal(T val) override
    {
        for (size_t i = 0; i < length; i++)
        {
            data[i] = val;
        }        
    }

    /// @brief Выводит в консоль сведения об объекте
    void Print() const override
    {
        std::cout << "VectorRam object description:" << std::endl;
        std::cout << "type name: " << typeid(this).name() << std::endl;
        std::cout << "address: " << this << std::endl;
        std::cout << "vector type: " << this->vectorType << std::endl;
        std::cout << "length: " << length << std::endl;        
        std::cout << "sizeof 1 data element: " << sizeof(T) << std::endl;
        std::cout << "size of data: " << sizeof(T) * Length() << std::endl;
        std::cout << "dataLocation: " << this->dataLocation << std::endl;
    }

    /// @brief Выводит в консоль элементы вектора в заданном диапазоне
    void PrintData(unsigned long long indStart,
        unsigned long long length) const override
    {
        if(indStart + length > Length())
        {
            throw std::runtime_error("Exception in PrintData()! Out of range: indStart + length > Length()");
        }

        std::string splitter = " ";
        if(this->vectorType == VectorType::VectorColumn)
            splitter = "\n";

        for (size_t i = indStart; i < length; i++)
        {
            std::cout << data[i] << splitter;
        }
        std::cout << std::endl;
    }

    /// @brief Выводит в консоль все элементы вектора
    void PrintData() const override
    {
        PrintData(0, Length());
    }


    /// @brief Возвращает длину вектора (количество элементов)
    /// @return 
    size_t Length() const override
    {
        return length;
    }

    /// @brief Возвращает значение элемента вектора, расположенного по указанному индексу
    T GetValue(unsigned long long index) const override
    {
        if (index >= Length())
            throw std::out_of_range("SetValue(): Index out of range!");

        return data[index];
    }

    /// @brief Устанавливает значение элемента вектора, расположенного по указанному индексу
    bool SetValue(unsigned long long index, T value) override
    {        
        if (index >= Length())
            return false;

        data[index] = value;
        return true;
    }

    /// @brief Очищает массивы данных и устанавливает размер вектора в 0
    void ClearData() override
    {
        delete[] data;
        data = nullptr;
        this->length = 0;
    }

};