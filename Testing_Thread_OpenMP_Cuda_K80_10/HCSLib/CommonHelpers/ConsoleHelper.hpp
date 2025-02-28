#pragma once

/// @brief Вспомогательный класс для работы с консолью
struct ConsoleHelper
{
    static void PrintLine(std::string string)
    {
        std::cout << string << std::endl;
    }

    static void WaitAnyKey(std::string message = "Press Enter to continue...")
    {
        std::cout << message;
        std::cout << std::flush;
        int ch = std::getchar();
        while(ch = std::getchar())
        {
            std::cout << "\nch: " << ch << "\n";
            if(ch == 10)
                break;
        };
        
    }

    template<typename T>
    static void PrintKeyValue(std::string key, T value,
        std::string splitter = ": ",
        bool isEndl = true)
    {
        std::cout << key << splitter << value;
        if(isEndl)
            std::cout << std::endl;
    }

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

    static bool GetBoolFromUser(std::string message, std::string errorMessage = "Error! Enter bool value (y, n, 0, 1)")
    {
        while (1)
        {
            try
            {
                std::cout << message;
                std::string userInput;
                std::cin >> userInput;
                
                if(userInput == "y" || userInput == "1")
                    return true;
                if(userInput == "n" || userInput == "0")
                    return false;
                std::cout << errorMessage << std::endl;
            }
            catch(const std::exception& e)
            {
                std::cout << errorMessage << std::endl;
            }
        }
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

    static double GetDoubleFromUser(std::string message = "Enter double value: ", std::string errorMessage = "Error! Enter double number")
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
                    double value = std::stod(userInput);
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