#pragma once

#include "../PrintParams.hpp"

/// @brief Параметры видеоадаптера
struct GpuParams
{
    /// @brief УИД GPU
    unsigned id{0};
    /// @brief Объём видеопамяти, доступной для вычислений, Гбайт
    unsigned VRamSizeGb{0};
    /// @brief Количество потоковых мультипроцессоров
    unsigned SmNumber{0};

    void Print(PrintParams pp)
    {
        std::cout << "GpuParams::Print(PrintParams pp)" << std::endl;
        std::cout << pp.startMes;
        std::cout << "id"           << pp.splitterKeyValue << id            << pp.splitter;
        std::cout << "VRamSizeGb"   << pp.splitterKeyValue << VRamSizeGb    << pp.splitter;
        std::cout << "SmNumber"     << pp.splitterKeyValue << SmNumber      << pp.splitter;
        std::cout << pp.endMes;
        if(pp.isEndl)
            std::cout << std::endl;
    }
};