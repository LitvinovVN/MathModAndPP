// Задача 01. Вывести в консоль текстовую строку "Hello World!"
// Запуск:
// gcc main.c -o app
// ./app

#include <stdio.h>              // подключаем заголовочный файл stdio.h (содержит определение функции printf)

void c_hello()                  // определяем функцию c_hello
{
    printf("Hello World!\n");   // выводим строку на консоль
}

int main()                      // определяем функцию main
{
    c_hello();                  // вызываем функцию c_hello

    return 0;
}