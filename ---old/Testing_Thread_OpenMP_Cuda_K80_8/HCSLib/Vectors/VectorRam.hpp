#pragma once

template<typename T>
class VectorRam : public Vector<T>
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

    size_t Size() const override
    {
        return size;
    }

};