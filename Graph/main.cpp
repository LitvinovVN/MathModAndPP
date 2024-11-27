// Графовая модель вычислительного процесса
// set PATH=%PATH%;C:\mingw64\bin
// g++ main.cpp -o app
// ./app

#include "Graph.hpp"

/// @brief Перечисление типов этапов вычислительного процесса
enum class CalcProcessStep
{
    Start,
    CreateVector,
    InitVector,
    ScalarProduct,
    End
};

/// @brief Перегрузка оператора << для перечисления CalcProcess
std::ostream &operator<<(std::ostream &os, CalcProcessStep step)
{
    switch (step)
    {
    case CalcProcessStep::Start:
        os << "Start";
        break;
    case CalcProcessStep::CreateVector:
        os << "CreateVector";
        break;
    case CalcProcessStep::InitVector:
        os << "InitVector";
        break;
    case CalcProcessStep::ScalarProduct:
        os << "ScalarProduct";
        break;
    case CalcProcessStep::End:
        os << "End";
        break;
    default:
        os << "!!! ";
        break;
    }

    return os;
}

////////////////////////////////////////////////////////////////

/// @brief Перечисление мест расположения данных
enum class DataLocation
{
    RAM,
    VRAM,
    RAM_VRAM
};

/// @brief Перегрузка оператора << для перечисления CalcProcess
std::ostream &operator<<(std::ostream &os, DataLocation dataLocation)
{
    switch (dataLocation)
    {
    case DataLocation::RAM:
        os << "RAM";
        break;
    case DataLocation::VRAM:
        os << "VRAM";
        break;
    case DataLocation::RAM_VRAM:
        os << "RAM_VRAM";
        break;
    default:
        os << "!!! ";
        break;
    }

    return os;
}

/////////////////////////////////////////////////////

struct Algorythm
{
    CalcProcessStep _calcProcessStep;

    Algorythm(CalcProcessStep calcProcessStep) : _calcProcessStep(calcProcessStep)
    {
    }
};

int main()
{
    std::cout << "---Graph<unsigned, std::string>---" << std::endl;
    Graph<unsigned, std::string> graph;
    graph.AddNode(0, "start");
    graph.AddNode(1, "create a", 0);
    graph.AddNode(2, "create b", 0);
    graph.AddNode(3, "init a", 1);
    graph.AddNode(4, "init b", 2);
    graph.AddNode(5, "(a, b)", {3, 4});
    graph.AddNode(6, "end", 5);
    graph.Print();

    std::cout << "---Copy---" << std::endl;
    Graph g2 = graph;
    g2.Print();

    /////////////////////////////

    std::cout << "---Graph<unsigned, std::string>---" << std::endl;
    Graph<unsigned, CalcProcessStep> graph3;
    graph3.AddNode(0, CalcProcessStep::Start);
    graph3.AddNode(1, CalcProcessStep::CreateVector, 0);
    graph3.AddNode(2, CalcProcessStep::CreateVector, 0);
    graph3.AddNode(3, CalcProcessStep::InitVector, 1);
    graph3.AddNode(4, CalcProcessStep::InitVector, 2);
    graph3.AddNode(5, CalcProcessStep::ScalarProduct, {3, 4});
    graph3.AddNode(6, CalcProcessStep::End, 5);
    graph3.Print();

    std::cout << "---Copy---" << std::endl;
    Graph g4 = graph3;
    g4.Print();

    /////////////////
    std::map<CalcProcessStep, std::vector<DataLocation>> calcProcessStepDataLocations;
    calcProcessStepDataLocations[CalcProcessStep::CreateVector] =
        {DataLocation::RAM, DataLocation::VRAM, DataLocation::RAM_VRAM};

    for (const auto &[key, value] : calcProcessStepDataLocations)
    {
        std::cout << key << " -> ";
        std::cout << "{ ";
        for (auto &el : calcProcessStepDataLocations[key])
        {
            std::cout << el << " ";
        }
        std::cout << "}" << std::endl;
    }
}