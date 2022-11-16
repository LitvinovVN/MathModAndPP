/* Задача 08. Ввести с консоли размеры двумерного массива.
 Создать двумерный массив.
 Инициализировать значения элементов двумерного массива индексами.
 Вывести двумерный массив в консоль

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

// Создаёт двумерный массив float в динамической памяти и возвращает на него указатель
float* Array2D_Create(int numRows, int numColumns)
{
    float* arr = (float*)malloc(numRows * numColumns * sizeof(float));
    return arr;
}

// Инициализирует элементы массива их индексами
void Array2D_InitByIndexes(float* fArray, int numRows, int numColumns)
{
    size_t j = 0;
    while(j < numRows)
    {
        size_t i = 0;
        while(i < numColumns)
        {
            int index = i + j * numColumns;
            fArray[index] = index;
            i++;
        }                
        j++;
    }
}

// Выводит элементы массива в консоль
void Array2D_Print(float* fArray, int numRows, int numColumns)
{
    size_t j = 0;
    while(j < numRows)
    {
        printf("%d: ", j);
        size_t i = 0;
        while(i < numColumns)
        {
            int index = i + j * numColumns;
            printf("%g ", fArray[index]);
            i++;
        }
        printf("\n");  
        j++;
    }         
}

//////////////////////////////////////////////////////////////////////////////

int main()
{    
    int numRows = inputIntNumber("Input number of array rows: ");
    int numColumns = inputIntNumber("Input number of array columns: ");

    float* fArray2D = Array2D_Create(numRows, numColumns);
    Array2D_InitByIndexes(fArray2D, numRows, numColumns);

    Array2D_Print(fArray2D, numRows, numColumns);

    return 0;
}