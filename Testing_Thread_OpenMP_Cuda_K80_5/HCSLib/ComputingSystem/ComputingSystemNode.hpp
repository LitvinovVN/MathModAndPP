#pragma once

#include <iostream>
#include <map>

#include "../PrintParams.hpp"
#include "GpuParams.hpp"

/// @brief Узел вычислительной системы
class ComputingSystemNode
{
    /// @brief УИД вычислительного узла
    unsigned id{0};

    /// @brief Количество потоков CPU, задействуемых в вычислениях
    unsigned threadsNum{0};

    /// @brief Максимальный объём RAM, доступной к задействованию в вычислениях
    unsigned RamSize{0};

    /// @brief Сведения о GPU, задействуемых в вычислениях
    std::map<unsigned, GpuParams> Gpus;

public:
    unsigned GetId() const
    {
        return id;
    }

    unsigned GetGpuNum() const
    {
        return Gpus.size();
    }

    bool IsGpuExists(unsigned id)
    {
        if(Gpus.count(id)>0)
            return true;

        return false;
    }

    bool AddGpu(GpuParams gpu)
    {
        if(IsGpuExists(gpu.id))
            return false;

        Gpus[gpu.id] = gpu;
        return true;
    }

    void Print(PrintParams pp)
    {
        std::cout << "ComputingSystemNode::Print(PrintParameters pp)" << std::endl;
        std::cout << pp.startMes;
        std::cout << "id"           << pp.splitterKeyValue << id            << pp.splitter;
        std::cout << "GPU number"   << pp.splitterKeyValue << GetGpuNum()   << pp.splitter;

        for(auto& gpu : Gpus)
        {
            gpu.second.Print(PrintParams{});            
        }

        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }

    
};