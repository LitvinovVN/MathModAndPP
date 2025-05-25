#pragma once


/// @brief Элемент меню
struct MenuCommandItem
{
    MenuCommand comm = MenuCommand::None;// Команда
    std::vector<std::string> keys;// Список ключей
    std::function<void()> func;// Вызываемая функция
    std::string desc;// Описание команды

    MenuCommandItem()
    {}

    MenuCommandItem(MenuCommand comm,
        std::vector<std::string> keys,
        std::function<void()> func,
        std::string desc)
            : comm(comm), keys(keys), func(func), desc(desc)
    {}

    void Reset()
    {
        comm = MenuCommand::None;
        keys = {};
        func = nullptr;
        desc = "Command not choosed!";
    }

    bool CheckKey(const std::string& str)
    {
        bool isKey = false;
        for(auto& key : keys)
        {
            if(key == str)
            {
                isKey = true;
                break;
            }
        }
        return isKey;
    }
};

