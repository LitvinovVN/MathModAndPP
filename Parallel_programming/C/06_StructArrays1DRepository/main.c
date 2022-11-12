/* Задача 07. Создать структуру, содержащую структуры Arrays1D
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
struct arraysRepository_t{
    int repositorySize;
    Arrays1D* data;
};

//Опрелеляем новый тип
typedef struct arraysRepository_t Arrays1DRepository;
//////////////////////////////////////////////////

void Array1D_Print(Array1D arr)
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

Array1D Array1D_Create(int size)
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

Arrays1D Arrays1D_Create(int numElements, int dataNumElements)
{
    Arrays1D arr = {numElements};
    arr.data = (Array1D*) malloc(numElements * sizeof(Array1D));

    int i = 0;

    while(i < arr.numElements)
    {
        arr.data[i] = Array1D_Create(dataNumElements);
        i++;
    }

    return arr;
}

void Arrays1D_Print(Arrays1D arr)
{
    printf("Arrays1D:\n");
    printf("\tnumElements = %d\n", arr.numElements);
    printf("\tdata:\n");

    int i = 0;

    while(i < arr.numElements)
    {        
        Array1D_Print(arr.data[i]);
        i++;
    }
    printf("\n");
}

//////////////////////////////////////////////////

Arrays1DRepository Arrays1DRepository_Create(int reposytorySize, int numElements, int dataNumElements)
{
    Arrays1DRepository repo={reposytorySize};
    repo.data = (Arrays1D*) malloc(reposytorySize * sizeof(Arrays1D));

    int i = 0;

    while(i < repo.repositorySize)
    {
        repo.data[i] = Arrays1D_Create(numElements, dataNumElements);
        i++;
    }

    return repo;

}
void Arrays1DRepository_Print(Arrays1DRepository repo)
{
    printf("Arrays1DRepository:\n");
    printf("\treposytorySize = %d\n", repo.repositorySize);
    printf("\tdata:\n");

    int i = 0;

    while(i < repo.repositorySize)
    {        
        Arrays1D_Print(repo.data[i]);
        i++;
    }
    printf("\n");
}

//////////////////////////////////////////////////

void main() { 
    Arrays1D arrays = Arrays1D_Create(3, 10);
    Arrays1D_Print(arrays);

    Array1D arr0 = Array1D_Create(3);
    Array1D arr1 = Array1D_Create(4);
    Array1D arr2 = Array1D_Create(5);
    Arrays1D_Insert(&arrays, arr0, 0);
    Arrays1D_Insert(&arrays, arr1, 1);
    Arrays1D_Insert(&arrays, arr2, 2);
    printf("------------------------\n");
    Arrays1D_Print(arrays);

    printf("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>\n");
    Arrays1DRepository repo0 = Arrays1DRepository_Create(2, 3, 4);
    Arrays1DRepository_Print(repo0);

    getch();
}