#pragma once

#include "DataTypeEnum.hpp"
#include "PrintParams.hpp"

struct DataType
{
    DataTypeEnum dataTypeEnum;

    DataType(DataTypeEnum dataTypeEnum) :
        dataTypeEnum(dataTypeEnum)
    {}

    static bool TryParse(DataType& dataType, std::string str)
    {
        if(str == "float")
        {
            dataType.dataTypeEnum = DataTypeEnum::dt_float;
            return true;
        }
        else if(str == "double")
        {
            dataType.dataTypeEnum = DataTypeEnum::dt_double;
            return true;
        }


        return false;
    }

    /// @brief Возвращает объём памяти, занимаемый одним элементом данного типа
    /// @return 
    unsigned GetSize() const
    {
        if (dataTypeEnum == DataTypeEnum::dt_float)
            return sizeof(float);
        else if (dataTypeEnum == DataTypeEnum::dt_double)
            return sizeof(double);

        throw std::runtime_error("DataType::GetSize(): DataTypeEnum not recognized!");
    }

    /// @brief Выводит в консоль сведения о типе данных
    /// @param pp 
    void Print(PrintParams pp = PrintParams{}) const
    {
        std::cout << pp.startMes;
        std::cout << dataTypeEnum;
        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};

std::ostream& operator<<(std::ostream& os, DataType dts)
{
    os << dts.dataTypeEnum;

    return os;
}