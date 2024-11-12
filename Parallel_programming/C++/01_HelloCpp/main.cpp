// Задача 01. Вывести в консоль текстовую строку "Hello World!"
// Запуск:
// g++ main.cpp -o app
// ./app

#include <iostream>              // подключаем заголовочный файл iostream (содержит определение std::cout)

void cpp_hello()                 // определяем функцию cpp_hello
{
    std::cout << "Hello World!" << std::endl; // выводим строку на консоль
}

int main()                      // определяем функцию main
{
    cpp_hello();                // вызываем функцию cpp_hello

    return 0;
}