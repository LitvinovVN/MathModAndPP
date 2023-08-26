#include <stdio.h>
#include <malloc.h>
#include <thread>

void print_execution_time(const char* message, std::chrono::system_clock::time_point start,
std::chrono::system_clock::time_point end, size_t data_size_b)
{    
    auto elapsed_mks = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double elapsed_mks_double = elapsed_mks.count();
    double data_size_mb = data_size_b / 1024 / 1024;
    double speed_mb_sek = data_size_mb * 1000 * 1000 / elapsed_mks_double;
    printf(message);
    printf(": %ld bytes (%.4f mb); ", data_size_b, data_size_mb);
    printf("%.3lf mks;", elapsed_mks_double);    
    printf("speed: %.2f mb/sec\n", speed_mb_sek);
}

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

// 2 массива записываются на диск 2 потоками в 2 файла
void fwrite_arr_double_test_2threads_2files(size_t N,
    const char* file_name_1, const char* file_name_2)
{
    size_t N_1 = N/2;
    size_t N_2 = N/2;
    size_t data_size_b = N * sizeof(double); 

    double* data_1 = create_arr_double(N_1);
    double* data_2 = create_arr_double(N_2);

    for(int i=0; i<N_1; i++) data_1[i] = i + 0.123456789;    
    //print_arr_double(N_1, data_1);

    for(int i=0; i<N_2; i++) data_2[i] = -i - 0.123456789;    
    //print_arr_double(N_2, data_2);

    auto start_chrono_file_w = std::chrono::high_resolution_clock::now();
    std::thread th_fw_1(fwrite_arr_double, file_name_1, N_1, data_1);
    std::thread th_fw_2(fwrite_arr_double, file_name_2, N_2, data_2);
    th_fw_1.join();
    th_fw_2.join();
    auto end_chrono_file_w = std::chrono::high_resolution_clock::now();    
    print_execution_time("Writing test", start_chrono_file_w, end_chrono_file_w, data_size_b);   

    arr_double_fill_zeros(N_1, data_1);
    //print_arr_double(N_1, data_1);
    arr_double_fill_zeros(N_2, data_2);    
    //print_arr_double(N_2, data_2);
    
    auto start_chrono_file_r = std::chrono::high_resolution_clock::now();
    std::thread th_fr_1(fread_arr_double, file_name_1, N_1, data_1);
    std::thread th_fr_2(fread_arr_double, file_name_2, N_2, data_2);
    th_fr_1.join();
    th_fr_2.join();
    auto end_chrono_file_r = std::chrono::high_resolution_clock::now();    
    print_execution_time("Reading test", start_chrono_file_r, end_chrono_file_r, data_size_b);

    //print_arr_double(N_1, data_1);
    //print_arr_double(N_2, data_2);
}

int main()
{
    printf("Starting...\n");
    
    const char* file_name_1 = "data_1.bin";
    //const char* file_name_2 = "data_2.bin";
    //const char* file_name_1 = "C:\\data\\data_1.bin";
    const char* file_name_2 = "C:\\data\\data_2.bin";
    
    int N = 100000000;
    
    printf("--- Test 01---\n");
    fwrite_arr_double_test_2threads_2files(N, file_name_1, file_name_2);

    printf("--- Test 02---\n");
    fwrite_arr_double_test_2threads_2files(N, "test_2_1.bin", "test_2_2.bin");

    
}