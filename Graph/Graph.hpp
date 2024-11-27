#pragma once

#include <iostream>
#include <map>
#include <vector>

/// @brief Класс "Вершина графа"
struct Node
{

};

/// @brief Класс "Ребро графа"
struct Edge
{

};

/// @brief Класс "Граф"
/// @tparam T1 Тип ключа вершины графа
/// @tparam T2 Тип вершины графа
template<typename T1, typename T2>
class Graph
{
    /// @brief Вершины графа
    std::map<T1, T2> _nodes;
    /// @brief Список смежности
    std::map<T1, std::vector<T1>> _adgList;

public:
    Graph()
    {
        
    }

    void AddNode(T1 key, T2 value)
    {
        _nodes[key] = value;
    }

    void AddNode(T1 key, T2 value, T1 keyPrev)
    {
        _nodes[key] = value;
        _adgList[keyPrev].push_back(key);
    }

    void AddNode(T1 key, T2 value, std::initializer_list<T1> keys)
    {
        _nodes[key] = value;

        for(auto& keyPrev : keys)
        {
            _adgList[keyPrev].push_back(key);
        }
    }

    void Print()
    {
        std::cout << "Graph" << std::endl;
        for (const auto& [key, value] : _nodes)
        {
            std::cout << key << " (" << value << ") -> ";
            std::cout << "{ ";
            for(auto& node : _adgList[key])
            {
                std::cout << node << " ";
            }
            std::cout << "}" << std::endl;
        }
            
    }
};