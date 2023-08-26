#include <stdio.h>
#include <malloc.h>
#include <thread>

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
    
    const char* file_name_1 = "data_1.bin";
    const char* file_name_2 = "data_2.bin";
    
    int N = 10;    
    double* data_1 = create_arr_double(N);
    double* data_2 = create_arr_double(N);

    for(int i=0; i<N; i++) data_1[i] = i + 0.123456789;    
    print_arr_double(N, data_1);

    for(int i=0; i<N; i++) data_2[i] = -i - 0.123456789;    
    print_arr_double(N, data_2);

    std::thread th_fw_1(fwrite_arr_double, file_name_1, N, data_1);
    std::thread th_fw_2(fwrite_arr_double, file_name_2, N, data_2);
    th_fw_1.join();
    th_fw_2.join();

    arr_double_fill_zeros(N, data_1);    
    print_arr_double(N, data_1);

    arr_double_fill_zeros(N, data_2);    
    print_arr_double(N, data_2);
    
    std::thread th_fr_1(fread_arr_double, file_name_1, N, data_1);
    std::thread th_fr_2(fread_arr_double, file_name_2, N, data_2);
    th_fr_1.join();
    th_fr_2.join();

    print_arr_double(N, data_1);
    print_arr_double(N, data_2);
}