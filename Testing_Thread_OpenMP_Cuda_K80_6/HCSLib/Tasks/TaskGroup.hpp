#pragma once


/// @brief Группы задач (работа с векторами, матрицами ипр.)
enum class TaskGroup
{
    None,
    Vector,
    VecVec,
    Matrix,
    MatVec,
    VecMat,
    MatMat
};

std::ostream& operator<<(std::ostream& os, TaskGroup tg)
{
    switch (tg)
    {
    case TaskGroup::None:
        os << "None";
        break;
    case TaskGroup::Vector:
        os << "Vector";
        break;
    case TaskGroup::VecVec:
        os << "VecVec";
        break;
    case TaskGroup::Matrix:
        os << "Matrix";
        break;
    case TaskGroup::MatVec:
        os << "MatVec";
        break;
    case TaskGroup::VecMat:
        os << "VecMat";
        break;
    case TaskGroup::MatMat:
        os << "MatMat";
        break;
    
    default:
        break;
    }

    return os;
}