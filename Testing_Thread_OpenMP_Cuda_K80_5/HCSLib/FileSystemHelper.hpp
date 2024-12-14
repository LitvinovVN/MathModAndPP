#pragma once

//////////////// Файловая система (начало) ///////////////////

/// @brief Класс для работы с файловой системой
//#include <filesystem> // C++17
#include <fstream>
#include "sys/stat.h"
class FileSystemHelper
{
public:
    /// @brief Комбинирует имя папки и имя файла в путь к файлу
    /// @param dir_name 
    /// @param file_name 
    /// @return 
    static std::string CombinePath(const std::string& dir_name, const std::string& file_name)
    {
        return std::string{dir_name + "/" + file_name};
    }

    static std::string CombinePath()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;

        std::cout << "Enter file name: ";
        std::string fileName;
        std::cin >> fileName;

        std::string path = CombinePath(dirName, fileName);
        std::cout << "Path: " << path << std::endl;

        return path;
    }


    /// @brief Проверяет существование файла
    /// @return true - существует; false - не существует
    static bool IsFileExists(const std::string& path_file)
    {
        std::ifstream iff(path_file);
        //std::cout << "iff.good()" << iff.good() << std::endl;
        return iff.good();

        // C++17 
        /*if(std::filesystem::exists(path_file))
            return true;
        else
            return false;*/
    }

    /// @brief Проверяет существование файла
    /// @return true - существует; false - не существует
    static bool IsFileExists(const std::string& dir_name, const std::string& file_name)
    {
        auto filePath = CombinePath(dir_name, file_name);
        if(IsFileExists(filePath))
            return true;
        else
            return false;
    }

    static bool IsFileExists()
    {
        std::cout << "Enter file name: ";
        std::string fileName;
        std::cin >> fileName;
        bool isExists = IsFileExists(fileName);
        if(isExists)
            std::cout << "File exists (true)" << std::endl;
        else
            std::cout << "File not exists (false)" << std::endl;

        return isExists;
    }

    /// @brief Проверяет существование каталога
    /// @return true - существует; false - не существует
    static bool IsDirExists(const std::string& path_dir)
    {
        std::string filePath = CombinePath(path_dir,"tmp");
        std::ofstream fout(filePath,std::ios::app);
        bool isExists = fout.good();
        fout.close();
        if (isExists)// Удаляем временный файл
        {
            remove(filePath.c_str());
        }
        return isExists;
        // C++17   
        //if(std::filesystem::exists(path_dir))
        //    return true;
        //return false;
    }

    static bool IsDirExists()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;
        bool isExists = IsDirExists(dirName);
        if(isExists)
            std::cout << "Directory exists (true)" << std::endl;
        else
            std::cout << "Directory not exists (false)" << std::endl;

        return isExists;
    }


    /// @brief Создаёт каталог
    /// @return Результат создания нового каталога
    static bool CreateDir(const std::string& path_dir)
    {
        if(IsDirExists(path_dir))
            return false;

        int errCode = mkdir(path_dir.c_str(), S_IRWXU);
        bool result = !(bool)errCode;
        return result;
        //return std::experimental::create_directory(path_dir);
    }

    static bool CreateDir()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;
        bool res = CreateDir(dirName);
        if(res)
            std::cout << "Directory created (true)" << std::endl;
        else
            std::cout << "Directory not created (false)" << std::endl;

        return res;
    }


    static bool CreateFile(const std::string& dir_name, const std::string& file_name, const std::string& string_data)
    {
        //auto filePath = CombinePath(dir_name, file_name);
        std::string filePath = file_name;
        if(dir_name.size()>0)
            filePath = CombinePath(dir_name, file_name);
        std::cout << "filePath: " << filePath << std::endl;
        std::ofstream fout(filePath);
        if(string_data != "")
            fout << string_data;
        fout.close();

        return true;
    }

    static bool CreateFile()
    {
        std::cout << "Enter dir name (. - current dir): ";
        std::string dirName;
        std::cin >> dirName;

        std::cout << "Enter file name: ";
        std::string fileName;
        std::cin >> fileName;

        bool res = CreateFile(dirName, fileName, "");
        if(res)
            std::cout << "File created (true)" << std::endl;
        else
            std::cout << "File not created (false)" << std::endl;

        return res;
    }


    static bool RemoveFile(const std::string& dir_name, const std::string& file_name)
    {
        auto filePath = CombinePath(dir_name, file_name);
        int errCode = remove(filePath.c_str());

        return !(bool)errCode;
    }

    static bool RemoveFile()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;

        std::cout << "Enter file name: ";
        std::string fileName;
        std::cin >> fileName;

        bool res = RemoveFile(dirName, fileName);
        if(res)
            std::cout << "File removed (true)" << std::endl;
        else
            std::cout << "File not removed (false)" << std::endl;

        return res;
    }


    static bool RemoveDir(const std::string& dir_name)
    {        
        int errCode = remove(dir_name.c_str());

        return !(bool)errCode;
    }

    static bool RemoveDir()
    {
        std::cout << "Enter dir name: ";
        std::string dirName;
        std::cin >> dirName;
        
        bool res = RemoveDir(dirName);
        if(res)
            std::cout << "Directory removed (true)" << std::endl;
        else
            std::cout << "Directory not removed (false)" << std::endl;

        return res;
    }
};
/////////////////// Файловая система (конец) ///////////////////

