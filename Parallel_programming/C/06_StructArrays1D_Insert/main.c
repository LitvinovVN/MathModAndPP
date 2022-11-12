/* Задача 06. Добавить метод, добавляющий в Arrays1D структуру Array1D по указанному индексу

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

void Arrays1D_Insert(Arrays1D* arrays, Array1D arr, size_t index)
{
    arrays->data[index] = arr;
}

//////////////////////////////////////////////////

Arrays1D createArrays1D(int numElements, int dataNumElements)
{
    Arrays1D arr = {numElements};
    arr.data = (Array1D*) malloc(numElements * sizeof(Array1D));

    int i = 0;

    while(i < arr.numElements)
    {
        arr.data[i] = createArray1D(dataNumElements);
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
    Arrays1D arrays = createArrays1D(3, 10);
    printArrays1D(arrays);

    Array1D arr0 = createArray1D(3);
    Array1D arr1 = createArray1D(4);
    Array1D arr2 = createArray1D(5);
    Arrays1D_Insert(&arrays, arr0, 0);
    Arrays1D_Insert(&arrays, arr1, 1);
    Arrays1D_Insert(&arrays, arr2, 2);
    printf("------------------------\n");
    printArrays1D(arrays);

    getch();
}