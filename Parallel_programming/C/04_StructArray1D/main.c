/* Задача 04. Создать структуру "Динамический одномерный массив", содержащий элементы типа float.
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

void main() {    
    Array1D arr = createArray1D(10);
    arr.data[0] = 0.01;
    arr.data[9] = 0.09;
    printArray1D(arr);
   

    getch();
}