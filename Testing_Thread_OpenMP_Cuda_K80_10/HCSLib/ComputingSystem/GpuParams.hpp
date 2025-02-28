#pragma once

#include "../CommonHelpers/PrintParams.hpp"

/// @brief Параметры видеоадаптера
struct GpuParams
{
    /// @brief УИД GPU
    unsigned id{0};
    /// @brief Наименование GPU
    std::string name{""};
    /// @brief Объём видеопамяти, доступной для вычислений, Гбайт
    unsigned VRamSizeGb{0};
    /// @brief Количество потоковых мультипроцессоров
    unsigned SmNumber{0};
    /// @brief Пиковая пропускная способность видеопамяти, Гб/с
    double PeakMemoryBandwidthGbS{0};

    void Print(PrintParams pp)
    {
        //std::cout << "GpuParams::Print(PrintParams pp)" << std::endl;
        std::cout << pp.startMes;

        std::cout << "id"                       << pp.splitterKeyValue << id;
        std::cout << pp.splitter;
        std::cout << "name"                     << pp.splitterKeyValue << name;
        std::cout << pp.splitter;
        std::cout << "VRamSizeGb"               << pp.splitterKeyValue << VRamSizeGb;
        std::cout << pp.splitter;
        std::cout << "SmNumber"                 << pp.splitterKeyValue << SmNumber;
        std::cout << pp.splitter;
        std::cout << "PeakMemoryBandwidthGbS"   << pp.splitterKeyValue << PeakMemoryBandwidthGbS;

        std::cout << pp.endMes;

        if(pp.isEndl)
            std::cout << std::endl;
    }
};