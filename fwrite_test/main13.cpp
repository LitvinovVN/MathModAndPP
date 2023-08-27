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
}

// 1 массив записывается на диск 1 потоком в 1 файл
void fwrite_arr_double_test_1thread_1file(size_t N,
    const char* file_name)
{    
    size_t data_size_b = N * sizeof(double); 

    double* data = create_arr_double(N);

    for(int i=0; i<N; i++) data[i] = i + 0.123456789;    
    //print_arr_double(N, data);
    
    auto start_chrono_file_w = std::chrono::high_resolution_clock::now();
    std::thread th_fw_1(fwrite_arr_double, file_name, N, data);    
    th_fw_1.join();
    auto end_chrono_file_w = std::chrono::high_resolution_clock::now();    
    print_execution_time("Writing test", start_chrono_file_w, end_chrono_file_w, data_size_b);    
}

int main()
{
    printf("Starting...\n");       
    
    int N = 100000000;          

    printf("Test 01: Writing on same disk by 1 thread\n");
    fwrite_arr_double_test_1thread_1file(N, "test_1.bin");

    printf("Test 02: Writing on other disk by 1 thread\n");
    fwrite_arr_double_test_1thread_1file(N, "C:\\data\\test_2.bin");

    printf("Test 03: Writing on same disk by 2 threads on 2 files\n");
    fwrite_arr_double_test_2threads_2files(N, "test_3_1.bin", "test_3_2.bin");

    printf("Test 04: Writing on other disk by 2 threads on 2 files\n");
    fwrite_arr_double_test_2threads_2files(N, "C:\\data\\test_4_1.bin", "C:\\data\\test_4_2.bin");

    printf("Test 05: Writing on 2 different disks by 2 threads on 1 file\n");
    fwrite_arr_double_test_2threads_2files(N, "test_5_1.bin", "C:\\data\\test_5_2.bin");
}