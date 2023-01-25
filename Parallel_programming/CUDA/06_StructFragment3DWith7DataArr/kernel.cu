/* Задача 06. Array3D

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
#include "Array3D.c"
///////////////////////////////////////////////////////



int main()
{    
    int f_nx = 10;
    int f_ny = 5;
    int f_nz = 3;

    Array3D arr3d_RAM = Array3D_RAM_Create(f_nx, f_ny, f_nz);
    Array3D_RAM_InitByIndexes(&arr3d_RAM, DataType::R);
    //Array3D_Print(&arr3d_RAM, DataType::R);
    Array3D* arr3d_GPU = Array3D_GPU_Create_From_Array3D_RAM(arr3d_RAM);
    CudaArray3D_GPU_Print<<<1,1>>>(arr3d_GPU, DataType::R);

    
    // Массив фрагментов
    /*int numElements = 3;
    Array3D* arr3d_arr = (Array3D*)malloc(numElements*sizeof(Array3D));
    for(int i = 0; i < numElements; i++)
    {
        arr3d_arr[i] = Array3D_RAM_Create(f_nx, f_ny, f_nz);
    }
    Array3D_Array_Print(arr3d_arr, numElements, C0);*/

    printf("Application normal end\n");
    return 0;
}