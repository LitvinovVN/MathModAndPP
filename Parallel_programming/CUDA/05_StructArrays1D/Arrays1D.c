///////////////////////////////////////////////////////
// Создаём структуру "Одномерный массив элементов типа Array1D"
struct arrays_t {
    int numElements;
    Array1D* data;
};

//Определяем новый тип
typedef struct arrays_t Arrays1D;

//////////////////////////////////////////////////

Arrays1D Arrays1D_Create(int numElements)
{
    Arrays1D arr = {numElements};
    arr.data = (Array1D*) malloc(numElements * sizeof(Array1D));

    int i = 0;

    while(i < arr.numElements)
    {
        //arr.data[i] = Array1D_RAM_Create(5);
        i++;
    }

    return arr;
}

// Размещает Array1D в Arrays1D по указанному индексу index
void Arrays1D_AddArray1D(Arrays1D* arrays1D, int index, Array1D array1D)
{    
    arrays1D->data[index] = array1D;
}

// Выводит содержимое структуры Arrays1D arr в консоль
__host__ __device__
void Arrays1D_Print(Arrays1D arr)
{
    printf("Arrays1D:\n");
    printf("\tnumElements = %d\n", arr.numElements);
    printf("\tdata:\n");

    int i = 0;

    while(i < arr.numElements)
    {
        printf("\tIndex: %d\n", i);
        Array1D_Print(&arr.data[i]);
        i++;
    }
    printf("\n");
}

//////////////////////////////////////////////////