/* Задача 03_04. Добавить функцию Array1D_GPU_Create_From_Array1D_RAM,
 создающую в GPU структуру Array1D на основе структуры Array1D, размещённой в ОЗУ

 Запуск:
 nvcc kernel.cu -o app.exe
 ./app

*/

#include <stdio.h>
#include <stdlib.h> // Для использования malloc
#include "utils.c"
#include "floatArray.c"
#include "Array1D.c"
#include "Arrays1D.c"
///////////////////////////////////////////////////////



int main()
{
    // Запрашиваем у пользователя количество элементов Array1D,
    // расположенных в структуре Arrays1D
    //int numElements = IntNumber_Input("Input number of array elements: ");
    int numElements = 2;
    printf("numElements = %d\n", numElements);

    // Создаём в ОЗУ структуру Array1D, содержащую массив из 5 элементов
    Array1D array1D_RAM = Array1D_RAM_Create(5);
    // Инициализируем элементы массива, расположенного в структуре array1D_RAM,
    // значениями их индексов
    Array1D_RAM_InitByIndexes(array1D_RAM);

    // Создаём в ОЗУ структуру Arrays1D, содержащую массив из numElements
    // структур Array1D
    Arrays1D arrays1D_RAM = Arrays1D_Create(numElements);
    Arrays1D_AddArray1D(&arrays1D_RAM, 0, Array1D_RAM_Create(10));
    Arrays1D_AddArray1D(&arrays1D_RAM, 1, array1D_RAM);
    //Arrays1D_Print(arrays1D_RAM);

    /////////////////////////
    Arrays1D* arrays1D_GPU = Arrays1D_GPU_Create_From_Arrays1D_RAM(arrays1D_RAM);
    CudaArrays1D_GPU_Print<<<1,1>>>(arrays1D_GPU);

    // Освобождаем память
    Arrays1D_RAM_Destruct(&arrays1D_RAM);
    Arrays1D_GPU_Destruct(arrays1D_GPU);

    return 0;
}