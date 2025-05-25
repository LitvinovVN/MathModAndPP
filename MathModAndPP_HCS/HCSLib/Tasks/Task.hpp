#pragma once


/// @brief Задача (копирование, суммирование и пр.)
enum class Task
{
    None,
    Init,// Инициализация
    Copy,// Копирование
    Sum, // Суммирование
    Min, // Минимум
    Max  // Максимум
};

std::ostream& operator<<(std::ostream& os, Task tg)
{
    switch (tg)
    {
    case Task::None:
        os << "None";
        break;
    case Task::Init:
        os << "Init";
        break;
    case Task::Copy:
        os << "Copy";
        break;
    case Task::Sum:
        os << "Sum";
        break;
    case Task::Min:
        os << "Min";
        break;
    case Task::Max:
        os << "Max";
        break;
    default:
        break;
    }

    return os;
}