/* Задача 02. Ввести с консоли количество элементов массива.
 Создать массив.
 Инициализировать значения элементов массива индексами.
 Вывести массив в консоль

 Запуск:
 gcc main.c -o app
 ./app

*/

#include <stdio.h>
#include <stdlib.h> // Для использования malloc

// Получает строку-сообщение message, выводит её в консоль.
// Считывает целое число, введённое пользователем и возвращает его.
int inputIntNumber(const char message[])
{
    int numElements;
    printf(message);
    scanf("%d", &numElements);

    return numElements;
}

// Создаёт массив float в динамической памяти и возвращает на него указатель
float* createFloatArray(int numElements)
{
    float* arr = (float*)malloc(numElements * sizeof(float));
    return arr;
}

// Инициализирует элементы массива их индексами
void initFloatArrayByIndexes(float* fArray, int numElements)
{
    size_t i = 0;
    while(i < numElements)
    {
        fArray[i] = i;
        i++;
    }        
}

// Выводит элементы массива в консоль
void printFloatArrayByIndexes(float* fArray, int numElements)
{
    size_t i = 0;
    while(i < numElements)
    {
        printf("%g ",fArray[i]);
        i++;
    }        
}

//////////////////////////////////////////////////////////////////////////////

int main()
{    
    int numElements = inputIntNumber("Input number of array elements: ");
    printf("numElements = %d\n", numElements);

    float* fArray = createFloatArray(numElements);
    initFloatArrayByIndexes(fArray, numElements);

    printFloatArrayByIndexes(fArray, numElements);

    return 0;
}