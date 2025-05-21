#pragma once

#include "../Scalars/_IncludeScalars.hpp"
#include "_IncludeVectors.hpp"

/// @brief Вспомогательный класс для работы с векторами через интерфейс IVector
struct IVectorHelper
{    
    /// @brief Создаёт новый вектор, склеивая векторы-аргументы
    /// @tparam T 
    /// @param v1 
    /// @param v2 
    /// @return 
    template<typename T>
    static
    IVector<T>* Split(IVector<T>* v1, IVector<T>* v2,
        DataLocation newVectorDataLocation = DataLocation::RAM)
    {
        size_t resultVectorLength = v1->Length() + v2->Length();
        IVector<T>* IVectorSplitResultPtr = nullptr;

        // Если результирующий вектор должен располагпться в RAM
        if(newVectorDataLocation == DataLocation::RAM)
        {
            IVectorSplitResultPtr = new VectorRam<T>(resultVectorLength);

            if( v1->dataLocation == DataLocation::RAM &&
                v2->dataLocation == DataLocation::RAM)
            {
                size_t i = 0;

                for(size_t v1i = 0; v1i < v1->Length(); v1i++)
                {
                    auto value = v1->GetValue(v1i);
                    IVectorSplitResultPtr->SetValue(i, value);
                    i++;
                }

                for(size_t v2i = 0; v2i < v2->Length(); v2i++)
                {
                    auto value = v2->GetValue(v2i);
                    IVectorSplitResultPtr->SetValue(i, value);
                    i++;
                }

                return IVectorSplitResultPtr;
            }

            
        }

        throw std::runtime_error("IVectorHelper::Split(): Not realized!");
    }

    /// Вычисляет скалярное произведение векторов 
    template<typename T>
    static
    IScalar<T>* Dot(IVector<T>* v1, IVector<T>* v2,
        DataLocation resultDataLocation = DataLocation::RAM)
    {
        IScalar<T>* resultPtr = nullptr;

        // Если результирующий объект должен располагаться в RAM
        if(resultDataLocation == DataLocation::RAM)
        {
            resultPtr = new ScalarRam<T>();

            if( v1->dataLocation == DataLocation::RAM &&
                v2->dataLocation == DataLocation::RAM)
            {
                size_t i = 0;

                
                return resultPtr;
            }

            
        }

        throw std::runtime_error("IVectorHelper::Sum(): Not realized!");
    }
};