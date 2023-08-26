#include <stdio.h>
#include <malloc.h>

double* create_arr_double(size_t size)
{
    double* data = (double*)malloc(size*sizeof(double));
    return data;
}    

void print_arr_double(int size, double* arr)
{
    for(int i=0; i<size; i++)
        printf("%f ", arr[i]);
    printf("\n");
}

void arr_double_fill_zeros(int size, double* arr)
{
    for(int i=0; i<size; i++)
        arr[i] = 0;
}
    

int fwrite_arr_double(const char * file_name, int size, double* arr)
{
    FILE *fp;
    if( (fp = fopen(file_name,"wb")) == NULL)
    {
        printf("Cannot open file.\n");
        return 1;
    }
    fwrite(arr, sizeof(double), size, fp);    
    fclose(fp);

    return 0;
}

int fread_arr_double(const char * file_name, int size, double* arr)
{
    FILE *fp;
    if( (fp = fopen(file_name,"rb")) == NULL)
    {
        printf("Cannot open file.\n");
        return 1;
    }
    fread(arr, sizeof(double), size, fp);
    fclose(fp);

    return 0;
}

int main()
{
    printf("Starting...\n");
    
    int N = 10;    
    double* data = create_arr_double(N);
    for(int i=0; i<N; i++) data[i] = i;
    
    print_arr_double(N, data);
    fwrite_arr_double("data.bin", N, data);

    arr_double_fill_zeros(N, data);    
    print_arr_double(N, data);

    fread_arr_double("data.bin", N, data);    
    print_arr_double(N, data);
}