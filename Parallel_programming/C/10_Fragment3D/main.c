/* Задача 10. Ввести с консоли размеры трехмерного массива.
 Создать структуру fragment3d ("Трехмерный фрагмент"), описывающую трехмерный массив.
 Инициализировать значения элементов трехмерного массива индексами.
 Вывести трехмерный массив в консоль

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

// Создаёт трехмерный массив float в динамической памяти и возвращает на него указатель
float* array3d_create(int numRows, int numColumns, int laier)
{
    float* arr = (float*)malloc(numRows * numColumns * laier * sizeof(float));
    return arr;
}

// Инициализирует элементы массива их индексами
void array3d_initByIndexes(float* fArray, int numRows, int numColumns, int numLayer)
{
    size_t k = 0;
    while(k < numLayer)
    {
        size_t j = 0;
        while(j < numRows)
        {
            size_t i = 0;
            while(i < numColumns)
            {
                int index = i + j * numColumns + k * numRows * numColumns;
                fArray[index] = index;
                i++;
            }                
            j++;
        }
        k++;
    }
}

// Выводит элементы массива в консоль
void array3d_print(float* fArray, int numRows, int numColumns, int numLayer)
{
    size_t k = 0;
    while(k < numLayer)
    {
        printf("Layer %d:\n", k);
        size_t j = 0;
        while(j < numRows)
        {
            printf("%d: ", j);
            size_t i = 0;
            while(i < numColumns)
            {
                int index = i + j * numColumns + k * numRows * numColumns;
                printf("%g ", fArray[index]);
                i++;
            }
            printf("\n");
            j++;
        }
        printf("\n");
        k++;       
    }
    
}
//////////////////////////////////////////////////////////////////////////////

struct fragment3d_t {
    int numX;
    int numY;
    int numZ;
    float* data;
};

//Определяем новый тип
typedef struct fragment3d_t Fragment3D;

void Fragment3D_print(Fragment3D fragment)
{
    printf("Fragment3D:\n");
    printf("numX = %d\n", fragment.numX);
    printf("numY = %d\n", fragment.numY);
    printf("numY = %d\n", fragment.numZ);
    printf("data = \n");
    
    array3d_print(fragment.data, fragment.numY, fragment.numX, fragment.numZ);
    printf("\n");
}

Fragment3D Fragment3D_create(int numX, int numY, int numZ)
{
    Fragment3D fragment = {numX, numY, numZ};
    fragment.data = (float*) malloc(numX * numY * numZ * sizeof(float));

    int k = 0;
    while(k < numZ)
        {
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
        k++;
        }
      
    return fragment;
}

//////////////////////////////////////////////////////////////////////////////
//Инициализирует фрагмент
void Fragment3D_initByIndexes(Fragment3D fragment3D)
{
    array3d_initByIndexes(fragment3D.data, fragment3D.numY, fragment3D.numX, fragment3D.numZ);
}

//////////////////////////////////////////////////////////////////////////////

int main()
{    
    int numRows = inputIntNumber("Input number of array rows: ");
    int numColumns = inputIntNumber("Input number of array columns: ");
    int numLayers = inputIntNumber("Input number of array layers: ");
    Fragment3D fragment3D = Fragment3D_create(numRows, numColumns, numLayers);
    Fragment3D_initByIndexes(fragment3D);

    Fragment3D_print(fragment3D);

    return 0;
}