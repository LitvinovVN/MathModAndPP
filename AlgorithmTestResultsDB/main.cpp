#include <iostream>

namespace TestResDb
{

/// @brief Группы типов операций
enum class TaskTypeGroup
{
    None   = 0,// Резерв (неинициализировано)
    Vector = 1,// Операции с одним вектором
    VecVec = 2,// Операции с двумя векторами
    Matrix = 3,// Операции с одной матрицей
    MatVec = 4,// Операции с матрицей и вектором
    MatMat = 5 // Операции с двумя матрицами
};

/// @brief Типы операций
enum class TaskType
{
    None  = 0, // Резерв (неинициализировано)
    // TaskTypeGroup::Vector = 1 // Операции с одним вектором
    V_Sum = 1, // Сумма элементов вектора
    V_Min = 2, // Минимальное значение элементов вектора
    V_Max = 3, // Максимальное значение элементов вектора

    // TaskTypeGroup::VecVec = 2 // Операции с двумя векторами
    VV_DotProduct = 4 // Скалярное произведение двух векторов
};

/// @brief Вычислительная задача
struct Task
{
    TaskTypeGroup taskTypeGroup = TaskTypeGroup::None;
    TaskType      taskType      = TaskType::None;

    Task(TaskTypeGroup taskTypeGroup, TaskType taskType)
        : taskTypeGroup(taskTypeGroup), taskType(taskType)
    {}

    void Print()
    {
        std::string taskTypeGroupName = "None";
        switch (taskTypeGroup)
        {
        case TaskTypeGroup::Vector:
            taskTypeGroupName = "Vector";
            break;
        
        default:
            break;
        }

        std::string taskTypeName = "None";
        switch (taskType)
        {
        case TaskType::V_Sum:
            taskTypeName = "V_Sum";
            break;
        
        default:
            break;
        }

        std::cout << "[ group: " << taskTypeGroupName
                  << ", type: "  << taskTypeName << "]" << std::endl;
    }
};

}



int main()
{
    std::cout << "---" << std::endl;
    std::cout << (int)TestResDb::TaskType::V_Sum << std::endl;

    TestResDb::Task task(TestResDb::TaskTypeGroup::Vector,
                         TestResDb::TaskType::V_Sum);
    task.Print();

    // Запрос к БД
    // TestParameters testParameters;
    // TestResDb::Task task(TestResDb::TaskTypeGroup::Vector, TestResDb::TaskType::V_Sum)
    // testParameters.Task = task
    // ...
    // long testId = GetTestId(testParameters);
    // auto res = GetTestResult(testId);
}