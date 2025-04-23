#pragma once

template<typename T>
class ScalarRam : public IScalar<T>
{
public:
    T data;
    
    ScalarRam(T value = 0) : data(value)
    {
        
    }

    ~ScalarRam()
    {
        
    }
    
    /// @brief Выводит в консоль сведения об объекте
    void Print() const override
    {
        std::cout << "ScalarRam object description:" << std::endl;
        std::cout << "type name: " << typeid(this).name() << std::endl;
        std::cout << "address: " << this << std::endl;      
        std::cout << "sizeof data element: " << sizeof(T) << std::endl;
        std::cout << "dataLocation: " << this->dataLocation << std::endl;
    }

    
    /// @brief Возвращает значение
    T GetValue() const override
    {
        return data;
    }

    /// @brief Устанавливает значение
    bool SetValue(T value) override
    {
        data = value;
        return true;
    }

    /// @brief Очищает массивы данных и устанавливает размер вектора в 0
    void ClearData() override
    {
        
    }

};