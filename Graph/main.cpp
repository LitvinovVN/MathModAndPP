// Графовая модель вычислительного процесса
// set PATH=%PATH%;C:\mingw64\bin
// g++ main.cpp -o app
// ./app

#include "Graph.hpp"

enum class CalcProcess
{
    Start,
    CreateVector,
    InitVector,
    ScalarProduct,
    End
};

std::ostream& operator<<(std::ostream& os, CalcProcess calcProcess)
{
    os << "!!! ";

    return os;
}

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
    Graph<unsigned, CalcProcess> graph3;
    graph3.AddNode(0, CalcProcess::Start);
    graph3.AddNode(1, CalcProcess::CreateVector, 0);
    graph3.AddNode(2, CalcProcess::CreateVector, 0);
    graph3.AddNode(3, CalcProcess::InitVector, 1);
    graph3.AddNode(4, CalcProcess::InitVector, 2);
    graph3.AddNode(5, CalcProcess::ScalarProduct, {3, 4});
    graph3.AddNode(6, CalcProcess::End, 5);
    graph3.Print();

    std::cout << "---Copy---" << std::endl;
    Graph g4 = graph3;
    g4.Print();
    
}