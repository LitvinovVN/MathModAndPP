#pragma once

/// @brief Вспомогательный класс для работы с консолью
struct ConsoleHelper
{
    /// @brief Запрашивает у пользователя целое число
    /// @param message Сообщение для пользователя
    /// @param errorMessage Сообщение об ошибке
    /// @return Введённое пользователем число
    static std::string GetStringFromUser(std::string message)
    {
        std::cout << message;
        std::string userInput;
        if(char(std::cin.peek()) == '\n')
            std::cin.ignore();

        if (std::cin.fail()) 
        {
            std::cin.clear();
            std::cin.ignore(32767, '\n');
        }
        getline(std::cin, userInput);
        return userInput;
    }

    /// @brief Запрашивает у пользователя целое число
    /// @param message Сообщение для пользователя
    /// @param errorMessage Сообщение об ошибке
    /// @return Введённое пользователем число
    static int GetIntFromUser(std::string message, std::string errorMessage = "Error! Enter integer number")
    {
        while (1)
        {
            try
            {
                std::cout << message;
                std::string userInput;
                std::cin >> userInput;
                int value = std::stoi(userInput);
            
                return value;
            }
            catch(const std::exception& e)
            {
                std::cout << errorMessage << std::endl;
            }
        }
    }

    static unsigned GetUnsignedIntFromUser(std::string message, std::string errorMessage = "Error! Enter integer number")
    {
        while (1)
        {
            try
            {
                std::cout << message;
                std::string userInput;
                std::cin >> userInput;

                if(isdigit(userInput[0]))
                {
                    unsigned value = std::stoul(userInput);
                    return value;
                }
                std::cout << errorMessage << std::endl;
            }
            catch(const std::exception& e)
            {
                std::cout << errorMessage << std::endl;
            }
        }
    }

    static size_t GetUnsignedLongLongFromUser(std::string message, std::string errorMessage = "Error! Enter integer number")
    {
        while (1)
        {
            try
            {
                std::cout << message;
                std::string userInput;
                std::cin >> userInput;

                if(isdigit(userInput[0]))
                {
                    size_t value = std::stoull(userInput);
                    return value;
                }
                std::cout << errorMessage << std::endl;
            }
            catch(const std::exception& e)
            {
                std::cout << errorMessage << std::endl;
            }
        }
    }
};