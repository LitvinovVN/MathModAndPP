#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <stdio.h>
#include <cmath>

void fwrite_double_array(size_t indStart, size_t numElements, double* arr)
{
    printf("\nfwrite_double_array()\n");
    for (size_t i = indStart; i < numElements; i++)
    {
        printf("%lf ", arr[i]);
    }

    FILE *f = fopen("double_arr_w.bin", "wb");
    size_t countw = fwrite(arr, sizeof arr, numElements, f);
    fclose(f);

    printf("countw = %ld\n", countw);    
}

void fread_double_array(size_t indStart, size_t numElements, double* arr)
{
    FILE *f = fopen("double_arr_w.bin", "wb");
    size_t countr = fread(arr, sizeof(double), numElements, f);
    fclose(f);
    printf("countr = %ld\n", countr);
}


void test_fwrite_fread_double_array()
{
    printf("--- test_fwrite_fread_double_array() starting ---\n");
    size_t Nw = 10;
    double* arr_w = (double*)malloc(Nw * sizeof(double));
    for (size_t i = 0; i < Nw; i++)
    {
        arr_w[i] = i + 0.1*i;
    }
    // Проверка    
    for (size_t i = 0; i < Nw; i++)
    {
        printf("%lf ", arr_w[i]);
    }
    
    // Записываем массив на диск с замером времени
    auto start_chrono_file_w = std::chrono::high_resolution_clock::now();
    FILE *f = fopen("double_arr_w.bin", "wb");
    size_t countw = fwrite(arr_w, sizeof arr_w, Nw, f);
    fclose(f);
    //fwrite_double_array(0, Nw, arr_w);
    auto end_chrono_file_w = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w - start_chrono_file_w);
    std::cout << "The time of fwrite_double_array: " << elapsed_ms_file_w.count() << " ms\n";    
    /////////////////////////////////////////////

    delete[] arr_w;


    size_t Nr = Nw;
    double* arr_r = (double*)malloc(Nr * sizeof(double));
    // Считываем массив с диска с замером времени
    auto start_chrono_file_r = std::chrono::high_resolution_clock::now();
    fread_double_array(0, Nr, arr_r);
    auto end_chrono_file_r = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_r = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_r - start_chrono_file_r);
    std::cout << "The time of fread_double_array: " << elapsed_ms_file_r.count() << " ms\n";    
    /////////////////////////////////////////////

    // Проверка
    for (size_t i = 0; i < Nw; i++)
    {
        if ( fabs((i+0.1*i)-arr_r[i]) > 0.00001 )
            printf("error! i=%ld\n",i);
    }
    ////////////

    delete[] arr_r;


    /*auto start_chrono_file_w = std::chrono::high_resolution_clock::now();

    FILE *fpw = fopen("double_arr_w.bin", "wb");
    FILE *fpw2 = fopen("C:\\data\\double_arr_w2.bin", "wb");
    //FILE *fpw2 = fopen("double_arr_w2.bin", "wb");

    //size_t countw = _fwrite_nolock(arr_w, sizeof(double), Nw, fpw);
    //size_t countw2 = _fwrite_nolock(arr_w, sizeof(double), Nw, fpw2);

    size_t countw = fwrite(arr_w, sizeof(double), Nw, fpw);
    size_t countw2 = fwrite(arr_w, sizeof(double), Nw, fpw2);
    fclose(fpw);
    fclose(fpw2);
    
    auto end_chrono_file_w = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w - start_chrono_file_w);
    std::cout << "The time of file writing: " << elapsed_ms_file_w.count() << " ms\n";
    printf("wrote %zu elements out of %zu\n", countw,  Nw);
    delete[] arr_w;*/

    /*auto start_chrono_file_r = std::chrono::high_resolution_clock::now();
    size_t Nr = Nw;
    double* arr_r = (double*)malloc(Nr * sizeof(double));   // определяем буфер достаточной длины
    FILE *fpr = fopen("double_arr_w.bin", "rb");
    //size_t countr = _fread_nolock(arr_r, sizeof(double), Nr, fpr);
    size_t countr = fread(arr_r, sizeof(double), Nr, fpr);
    fclose(fpr); 
    printf("read %zu elements out of %d\n", countr,  Nr);    
    auto end_chrono_file_r = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_r = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_r - start_chrono_file_r);
    std::cout << "The time of file reading: " << elapsed_ms_file_r.count() << " ms\n";*/

    
    if(Nr <= 100)
    {
        for(int i=0;i<Nr;i++)
            printf("%.4lf ", arr_r[i]);
    }

    
    delete[] arr_r;

    exit(1);
}

void test_fwrite_fread_char_array()
{
    char strw[] = "Hello METANIT.COM";
    size_t Nw = sizeof(strw);
    FILE *fpw = fopen("data.bin", "wb");
    size_t countw = _fwrite_nolock(strw, sizeof strw[0], Nw, fpw);
    printf("wrote %zu elements out of %zu\n", countw,  Nw);
    fclose(fpw);

    size_t Nr = Nw;
    char strr[Nr];   // определяем буфер достаточной длины
    FILE *fpr = fopen("data.bin", "rb");
    size_t countr = _fread_nolock(strr, sizeof strr[0], Nr, fpr); 
    printf("read %zu elements out of %d\n", countr,  Nr);
    printf(strr);
    fclose(fpr);

    exit(1);
}

void test_fopen(std::string path, double h, int indStart, int size, double* data)
{        
    //char * filename = "users.dat";    
    FILE *fp = fopen("out11_test_fopen.txt", "w");
    if (!fp)
    {
        printf("Error occured while opening file\n");
        return;
    }
      
    for(size_t i = indStart; i < indStart+size; i++)
    {
        fprintf(fp, "%d %lf %lf\n", i, i*h, data[i]);
    }

    fclose(fp);
    printf("File has been written\n");
}


void write_to_file(std::string path, double h, int indStart, int size, double* data) {
    std::ofstream out(path);          // поток для записи в файл
    for (size_t i = indStart; i < indStart+size; i++)
    {
        out << i << " " << i*h << " " << data[i] << "\n";
    }
    out.close();
}

void config()
{
    setlocale(LC_ALL, "");
    std::wcout<<L"Уравнение теплопроводности 1D"<< std::endl;
}

int main()
{
    //test_fwrite_fread_char_array();
    test_fwrite_fread_double_array();
    //////////////
    config();

    // Количество пространственных узлов
    int N = 1000001;
    // Время моделирования, сек.
    double t_end = 60;
    // Толщина пластины, м.
    double L = 0.1;
    // Коэффициент теплопроводности материала пластины, Вт/(м гр.Цельс.)
    double lambda = 46;
    // Плотность материала пластины, кг/м.куб.
    double ro = 7800;
    // Теплоёмкость материала пластины, Дж/(кг. град.Цельс.)
    double c = 460;

    // Начальная температура пластины, град. Цельс.
    double T0 = 20;
    // Температура левой границы пластины, град. Цельс.
    double Tl = 300;
    // Температура правой границы пластины, град. Цельс.
    double Tr = 100;

    double h = L/(N-1);
    std::wcout<<L"h = "<< h << L" м." << std::endl;

    double tau = t_end/100;
    std::wcout<<L"tau = "<< tau <<L" сек." << std::endl;

    double* T = new double[N];
    double* alfa = new double[N];
    double* beta = new double[N];

    std::wcout<<L"Выделено памяти: "<< 3.0 * N * sizeof(double) /(1024 * 1024) << L" Мб." << std::endl;

    // Инициализируем поле температуры в начальный момент времени
    for (size_t i = 0; i < N; i++)
    {
        T[i] = T0;
    }
        
    T[0] = Tl;
    T[N-1] = Tr;

    /*for (size_t i = 0; i < N; i++)
    {
        std::wcout << T[i] << " ";
    }*/
        
    auto start_chrono = std::chrono::high_resolution_clock::now(); 
    ///////////// Решение ///////////
    double time = 0;
    while (time < t_end)
    {
        time += tau;

        alfa[0] = 0;
        beta[0] = Tl;

        for (size_t i = 1; i < N-1; i++)
        {
            double ai = lambda / (h*h);
            double bi = 2 * lambda / (h*h) + ro * c / tau;
            double ci = lambda / (h*h);
            double fi = - ro * c * T[i] / tau;
            alfa[i] = ai / (bi - ci * alfa[i-1]);
            beta[i] = (ci * beta[i-1] - fi) / (bi - ci * alfa[i-1]);
        }
        
        for (size_t i = N-2; i > 0; i--)
        {
            T[i] = alfa[i] * T[i+1] + beta[i];
        }                
    }
    /////////// Окончание решения ///////////    
    auto end_chrono = std::chrono::high_resolution_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
    std::cout << "The time (chrono): " << elapsed_ms.count() << " ms\n";


    
    /*for (size_t i = 0; i < N; i++)
    {
        std::wcout << i*h << " " << T[i] << std::endl;
    }*/
    
    /*auto start_chrono_file_w1 = std::chrono::high_resolution_clock::now(); 
    std::ofstream out("out1.txt");          // поток для записи в файл
    for (size_t i = 0; i < N; i++)
    {
        out << i << " " << i*h << " " << T[i] << std::endl;
    }
    auto end_chrono_file_w1 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w1 - start_chrono_file_w1);
    std::cout << "The time (chrono): " << elapsed_ms_file_w1.count() << " ms\n";*/
    

    /*auto start_chrono_file_w3 = std::chrono::high_resolution_clock::now(); 
    write_to_file("out3.txt", h, 0, N, T);
    auto end_chrono_file_w3 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w3 - start_chrono_file_w3);
    std::cout << "The time (chrono): " << elapsed_ms_file_w3.count() << " ms\n";

    auto start_chrono_file_w4 = std::chrono::high_resolution_clock::now(); 
    write_to_file("C:\\data\\out4.txt", h, 0, N, T);
    auto end_chrono_file_w4 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w4 - start_chrono_file_w4);
    std::cout << "The time (chrono): " << elapsed_ms_file_w4.count() << " ms\n";*/
/*
    auto start_chrono_file_w5 = std::chrono::high_resolution_clock::now(); 
    std::thread tA1(write_to_file, "out5.txt", h, 0, N, T);  
    tA1.join();
    auto end_chrono_file_w5 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w5 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w5 - start_chrono_file_w5);
    std::cout << "The time (chrono): " << elapsed_ms_file_w5.count() << " ms\n";

    auto start_chrono_file_w6 = std::chrono::high_resolution_clock::now(); 
    std::thread tA2(write_to_file, "C:\\data\\out6.txt", h, 0, N, T);  
    tA2.join();
    auto end_chrono_file_w6 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w6 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w6 - start_chrono_file_w6);
    std::cout << "The time (chrono): " << elapsed_ms_file_w6.count() << " ms\n";


    auto start_chrono_file_w7 = std::chrono::high_resolution_clock::now(); 
    std::thread tA7_1(write_to_file, "out7_1.txt", h, 0, N/2, T);
    std::thread tA7_2(write_to_file, "out7_2.txt", h, N/2, N/2+1, T);
    //std::thread tA7_2(write_to_file, "C:\\data\\out7_2.txt", h, N/2, N/2+1, T);  
    tA7_1.join();
    tA7_2.join();
    auto end_chrono_file_w7 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w7 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w7 - start_chrono_file_w7);
    std::cout << "The time (chrono) 7: " << elapsed_ms_file_w7.count() << " ms (2 threads on 1 ssd)\n";

    auto start_chrono_file_w8 = std::chrono::high_resolution_clock::now(); 
    std::thread tA8_1(write_to_file, "out8_1.txt", h, 0, N/4, T);
    std::thread tA8_2(write_to_file, "out8_2.txt", h, N/4, N/4, T);
    std::thread tA8_3(write_to_file, "out8_3.txt", h, 2*N/4, N/4, T);
    std::thread tA8_4(write_to_file, "out8_4.txt", h, 3*N/4, N/4+1, T);
    //std::thread tA7_2(write_to_file, "C:\\data\\out7_2.txt", h, N/2, N/2+1, T);  
    tA8_1.join();
    tA8_2.join();
    tA8_3.join();
    tA8_4.join();
    auto end_chrono_file_w8 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w8 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w8 - start_chrono_file_w8);
    std::cout << "The time (chrono) 8: " << elapsed_ms_file_w8.count() << " ms (4 threads on 1 ssd)\n";

    auto start_chrono_file_w9 = std::chrono::high_resolution_clock::now(); 
    std::thread tA9_1(write_to_file, "out9_1.txt", h, 0, N/2, T);
    std::thread tA9_2(write_to_file, "C:\\data\\out9_2.txt", h, N/2, N/2+1, T);    
    tA9_1.join();
    tA9_2.join();
    auto end_chrono_file_w9 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w9 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w9 - start_chrono_file_w9);
    std::cout << "The time (chrono) 9: " << elapsed_ms_file_w9.count() << " ms (2 threads on 2 ssd)\n";
*/

    auto start_chrono_file_w10 = std::chrono::high_resolution_clock::now(); 
    std::thread tA10_1(write_to_file, "out10_1.txt", h, 0, N/4, T);
    std::thread tA10_2(write_to_file, "out10_2.txt", h, N/4, N/4, T);
    std::thread tA10_3(write_to_file, "C:\\data\\out10_3.txt", h, 2*N/4, N/4, T);
    std::thread tA10_4(write_to_file, "C:\\data\\out10_4.txt", h, 3*N/4, N/4+1, T);
    //std::thread tA7_2(write_to_file, "C:\\data\\out7_2.txt", h, N/2, N/2+1, T);  
    tA10_1.join();
    tA10_2.join();
    tA10_3.join();
    tA10_4.join();
    auto end_chrono_file_w10 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w10 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w10 - start_chrono_file_w10);
    std::cout << "The time (chrono) 10: " << elapsed_ms_file_w10.count() << " ms (4 threads on 2 ssd)\n";

    auto start_chrono_file_w11 = std::chrono::high_resolution_clock::now(); 
    test_fopen("-------", h, 0, N, T);
    auto end_chrono_file_w11 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms_file_w11 = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono_file_w11 - start_chrono_file_w11);
    std::cout << "The time (chrono) 11: " << elapsed_ms_file_w11.count() << " ms (test_fopen)\n";

    delete[] T;
    delete[] alfa;
    delete[] beta;

    return 0;
}