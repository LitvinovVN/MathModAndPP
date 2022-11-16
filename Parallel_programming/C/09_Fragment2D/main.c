/* Задача 09. Ввести с консоли размеры двумерного массива.
 Создать структуру fragment2d ("Двумерный фрагмент"), описывающую двумерный массив.
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

struct fragment2D_t {
    int numX;
    int numY;
    float* data;
};

//Определяем новый тип
typedef struct fragment2D_t Fragment2D;

void Fragment2D_Print(Fragment2D fragment)
{
    printf("Fragment2D:\n");
    printf("numX = %d\n", fragment.numX);
    printf("numY = %d\n", fragment.numY);
    printf("data = \n");
    
    Array2D_Print(fragment.data, fragment.numY, fragment.numX);
    printf("\n");
}

Fragment2D Fragment2D_Create(int numX, int numY)
{
    Fragment2D fragment = {numX, numY};
    fragment.data = (float*) malloc(numX * numY * sizeof(float));

    int j = 0;

    while(j < numY)
    {
        int i = 0;
        while(i < numX)
        {
            int index = i+j*numX;
            fragment.data[index] = 0;
            i++;
        }
        j++;
    }

    return fragment;
}

//////////////////////////////////////////////////////////////////////////////
//Инициализирует фрагмент
Fragment2D_InitByIndexes(Fragment2D fragment2D)
{
    Array2D_InitByIndexes(fragment2D.data, fragment2D.numY, fragment2D.numX);
}

//////////////////////////////////////////////////////////////////////////////

int main()
{    
    int numRows = inputIntNumber("Input number of array rows: ");
    int numColumns = inputIntNumber("Input number of array columns: ");

    Fragment2D fragment2D = Fragment2D_Create(numRows, numColumns);
    Fragment2D_InitByIndexes(fragment2D);

    Fragment2D_Print(fragment2D);

    return 0;
}