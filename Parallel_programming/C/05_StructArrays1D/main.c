/* Задача 05. Создать структуру "Динамический одномерный массив", содержащий элементы типа Array1D.
 Инициализировать поля структуры.
 Вывести структуру в консоль

 Запуск:
 gcc main.c -o app
 ./app

*/
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>

//////////////////////////////////////////////////
struct array_t {
    int size;
    float* data;
};

//Определяем новый тип
typedef struct array_t Array1D;
//////////////////////////////////////////////////
struct arrays_t {
    int numElements;
    Array1D* data;
};

//Определяем новый тип
typedef struct arrays_t Arrays1D;
//////////////////////////////////////////////////

void printArray1D(Array1D arr)
{
    printf("Array1D:\n");
    printf("\tsize = %d\n", arr.size);
    printf("\tdata = ");

    int i = 0;

    while(i < arr.size)
    {
        printf("%f ", arr.data[i]);
        i++;
    }
    printf("\n");
}

Array1D createArray1D(int size)
{
    Array1D arr = {size};
    arr.data = (float*) malloc(size * sizeof(float));

    int i = 0;

    while(i < arr.size)
    {
        arr.data[i] = 0;
        i++;
    }

    return arr;
}

//////////////////////////////////////////////////

Arrays1D createArrays1D(int numElements)
{
    Arrays1D arr = {numElements};
    arr.data = (Array1D*) malloc(numElements * sizeof(Array1D));

    int i = 0;

    while(i < arr.numElements)
    {
        arr.data[i] = createArray1D(5);
        i++;
    }

    return arr;
}

void printArrays1D(Arrays1D arr)
{
    printf("Arrays1D:\n");
    printf("\tnumElements = %d\n", arr.numElements);
    printf("\tdata:\n");

    int i = 0;

    while(i < arr.numElements)
    {        
        printArray1D(arr.data[i]);
        i++;
    }
    printf("\n");
}

//////////////////////////////////////////////////

void main() { 
    Arrays1D arrays = createArrays1D(3);
    printArrays1D(arrays);

    getch();
}