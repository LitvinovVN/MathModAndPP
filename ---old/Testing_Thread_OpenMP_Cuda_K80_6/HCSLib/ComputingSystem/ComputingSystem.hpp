#pragma once

#include <map>

/// @brief Вычислительная система
class ComputingSystem
{
    int id{0};// Идентификатор вычислительной системы
    std::string name{"TestSystem"}; // Наименование вычислительной системы
    std::string description{"TestSystem description"}; // Описание вычислительной системы
    std::string file_name{"ComputingSystem.txt"};// Имя файла с описанием вычислительной системы
    
    std::map<unsigned, ComputingSystemNode> nodes;// Вычислительные узлы

public:
    ComputingSystem()
    {}

    ComputingSystem(int id,
        std::string name,
        std::string description,
        std::string file_name = "ComputingSystem.txt"
        ) : id(id),
            name(name),
            description(description),
            file_name(file_name)
    {}

    /// @brief Добавляет вычислительный узел в вычислительную систему
    void AddNode(ComputingSystemNode node)
    {
        nodes[node.GetId()] = node;
    }

    /// @brief Выводит в консоль сведения о вычислительной системе
    void Print()
    {
        std::cout   << "Computing system details:"
                    << "\nid:          " << id
                    << "\nname:        " << name
                    << "\ndescription: " << description
                    << "\nfile_name:   " << file_name
                    << std::endl;
        std::cout   << "Nodes number: " << GetNodesNumber() << std::endl;
        
        for ( auto& node : nodes)
        {
            node.second.Print(PrintParams{"--- Node [", ": ", "; ", "] ---"});
        }
    }

    /// @brief Возвращает количество узлов вычислительной системы
    /// @return 
    unsigned GetNodesNumber()
    {
        unsigned cnt{0};

        /*for ( auto& node : nodes)
        {
            cnt++;
        }*/
        cnt = nodes.size();

        return cnt;
    }

    /// @brief Возвращает идентификатор вычислительной системы
    /// @return 
    int GetId() const
    {
        return id;
    }

    /// @brief Устанавливает идентификатор вычислительной системы
    /// @param id 
    void SetId(int id)
    {
        this->id = id;
    }

    /// @brief Записать сведения о вычислительной системе
    /// @param dir_name Каталог для записи
    /// @return 
    bool Serialize(const std::string& dir_name)
    {
        // Создаём каталог dir_name/id
        std::string path_dir = FileSystemHelper::CombinePath(dir_name, std::to_string(id));
        bool result = FileSystemHelper::CreateDir(path_dir);
        if (!result)
        {
            std::cerr << "Cannot create dir " << path_dir << std::endl;
            return false;
        }
        std::string data = std::to_string(id) + "\n" + name + "\n" + description + "\n";
        FileSystemHelper::CreateFile(path_dir, file_name, data);

        return true;
    }

    static ComputingSystem Deserialize(const std::string& dir_name,
                const int id,
                const std::string& file_name = "ComputingSystem.txt")
    {
        ComputingSystem computingSystem;

        std::string dir_Path = FileSystemHelper::CombinePath(dir_name,
                                std::to_string(id));
        std::string filePath = FileSystemHelper::CombinePath(dir_Path, file_name);
        std::ifstream fin(filePath);

        if(!fin.good())
            throw std::runtime_error("Error in opening file " + filePath);

        int f_id;
        std::string f_name;
        std::string f_description;

        //fin >> f_id;
        std::string f_id_str;
        std::getline(fin, f_id_str);
        f_id = std::stoi(f_id_str);
        if(f_id != id)
            throw std::runtime_error("Error in file " + filePath);

        //fin >> f_name;
        std::getline(fin, f_name);
        std::getline(fin, f_description);

        return ComputingSystem(f_id, f_name, f_description);
    }

    static ComputingSystem GetDataFromUser()
    {
        int id = ConsoleHelper::GetIntFromUser("Enter computing system id: ");
        std::string name = ConsoleHelper::GetStringFromUser("Enter computing system name: ");
        std::string description = ConsoleHelper::GetStringFromUser("Enter computing system description: ");

        return ComputingSystem(id, name, description);
    }
};

