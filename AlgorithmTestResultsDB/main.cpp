#include <iostream>
#include <map>

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
    Sum = 1, // Сумма элементов вектора
    Min = 2, // Минимальное значение элементов вектора
    Max = 3, // Максимальное значение элементов вектора

    // TaskTypeGroup::VecVec = 2 // Операции с двумя векторами
    DotProduct = 4 // Скалярное произведение двух векторов
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
        case TaskTypeGroup::VecVec:
            taskTypeGroupName = "VecVec";
            break;
        
        default:
            break;
        }

        std::string taskTypeName = "None";
        switch (taskType)
        {
        case TaskType::Sum:
            taskTypeName = "Sum";
            break;
        
        default:
            break;
        }

        std::cout << "[ group: " << taskTypeGroupName
                  << ", type: "  << taskTypeName << "]" << std::endl;
    }
};

}

/// @brief База данных результатов тестов производительности вычислительной системы
template<typename T>
class CompSystemPerfDb
{
    size_t lastId = 0;// Последний добавленный идентификатор
    std::map<size_t, T> _data;
public:
    /// @brief Добавляет элемент в словарь и возвращает его ключ
    /// @param entry 
    /// @return 
    size_t Add(T entry)
    {
        auto nextId = lastId + 1;
        while(_data.count(nextId))
            nextId++;

        _data[nextId] = entry;
        return nextId;
    }

    void Print()
    {
        for(auto& [key, value] : _data)
        {
            std::cout << "[" << key << ": " << value << "]" << std::endl;
        }
    }

};

/// @brief Запись в файле БД результатов эксперимента
struct FileDbRow
{
    size_t          id;// ID записи, 8 байт
    unsigned short  idCompSystem;//ID вычислительной системы, 2 байта
    unsigned short  idTaskTypeGroup;//ID группы задач, 2 байта
    unsigned short  idTaskType;//ID задачи, 2 байта
};

bool TestingCompSystemPerfDb()
{
    CompSystemPerfDb<std::string> db;
    db.Add("111");
    db.Add("222");
    db.Add("333");
    db.Add("444");
    db.Print();

    return true;
}

int main()
{
    std::cout << "---" << std::endl;
    std::cout << (int)TestResDb::TaskType::Sum << std::endl;

    TestResDb::Task task(TestResDb::TaskTypeGroup::Vector,
                         TestResDb::TaskType::Sum);
    task.Print();

    TestingCompSystemPerfDb();

    // Генерируем id    

    // Запрос к БД
    // TestParameters testParameters;
    // TestResDb::Task task(TestResDb::TaskTypeGroup::Vector, TestResDb::TaskType::V_Sum)
    // testParameters.Task = task
    // ...
    // long testId = GetTestId(testParameters);
    // auto res = GetTestResult(testId);
}