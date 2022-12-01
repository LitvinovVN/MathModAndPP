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
    int numElements = IntNumber_Input("Input number of array elements: ");
    printf("numElements = %d\n", numElements);

    Array1D array1D_RAM = Array1D_RAM_Create(numElements);
    Array1D_RAM_InitByIndexes(array1D_RAM);

    Arrays1D arrays = Arrays1D_Create(3);
    Arrays1D_AddArray1D(&arrays, 0, Array1D_RAM_Create(10));
    Arrays1D_AddArray1D(&arrays, 1, array1D_RAM);
    Arrays1D_Print(arrays);

    /////////////////////////
    Array1D* array1D_GPU = Array1D_GPU_Create_From_Array1D_RAM(array1D_RAM);
    CudaArray1D_GPU_Print<<<1,1>>>(array1D_GPU);

    // Освобождаем память
    Array1D_RAM_Destruct(&array1D_RAM);
    Array1D_GPU_Destruct(array1D_GPU);

    return 0;
}