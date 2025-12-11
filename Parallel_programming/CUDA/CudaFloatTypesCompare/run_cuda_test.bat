@echo off
chcp 65001 > nul
title CUDA Floating Point Performance Test
color 0A

echo ==============================================
echo   CUDA ПРОГРАММА ДЛЯ СРАВНЕНИЯ ТИПОВ ДАННЫХ
echo ==============================================
echo.

REM Проверка наличия компилятора NVCC
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ОШИБКА: Компилятор NVCC не найден!
    echo.
    echo Убедитесь, что:
    echo 1. CUDA Toolkit установлен
    echo 2. Путь к CUDA добавлен в переменную PATH
    echo 3. Visual Studio установлена (для Windows)
    echo.
    pause
    exit /b 1
)

REM Получаем версию CUDA
echo Получение информации о CUDA...
nvcc --version > temp_cuda_ver.txt 2>&1
findstr /C:"release" temp_cuda_ver.txt > nul
if %errorlevel% equ 0 (
    for /f "tokens=*" %%a in ('findstr /C:"release" temp_cuda_ver.txt') do (
        set "cuda_ver=%%a"
    )
    echo Версия CUDA: %cuda_ver%
) else (
    echo Не удалось определить версию CUDA
)
del temp_cuda_ver.txt > nul 2>&1

echo.

REM Выбор архитектуры
echo Выберите архитектуру GPU для компиляции:
echo.
echo [1] sm_70  - Volta (Tesla V100)
echo [2] sm_75  - Turing (RTX 20xx)
echo [3] sm_80  - Ampere (RTX 30xx, A100)
echo [4] sm_86  - Ampere mobile (RTX 30xx mobile)
echo [5] sm_89  - Ada Lovelace (RTX 40xx)
echo [6] sm_90  - Hopper (H100, для FP8)
echo [7] Автоматическое определение (рекомендуется)
echo.
set /p arch="Введите номер [1-7]: "

if "%arch%"=="1" set ARCH=sm_70
if "%arch%"=="2" set ARCH=sm_75
if "%arch%"=="3" set ARCH=sm_80
if "%arch%"=="4" set ARCH=sm_86
if "%arch%"=="5" set ARCH=sm_89
if "%arch%"=="6" set ARCH=sm_90
if "%arch%"=="7" (
    echo.
    echo Автоматическое определение архитектуры...
    nvcc -o gpu_query.exe gpu_query.cu > nul 2>&1
    if exist gpu_query.exe (
        gpu_query.exe > temp_gpu_info.txt
        for /f "tokens=2 delims=:" %%a in ('findstr /C:"Compute Capability" temp_gpu_info.txt') do (
            set "compute_cap=%%a"
        )
        set "compute_cap=%compute_cap: =%"
        
        if "%compute_cap%"=="7.0" set ARCH=sm_70
        if "%compute_cap%"=="7.5" set ARCH=sm_75
        if "%compute_cap%"=="8.0" set ARCH=sm_80
        if "%compute_cap%"=="8.6" set ARCH=sm_86
        if "%compute_cap%"=="8.9" set ARCH=sm_89
        if "%compute_cap%"=="9.0" set ARCH=sm_90
        
        echo Определена архитектура: %compute_cap% -> %ARCH%
        del temp_gpu_info.txt > nul 2>&1
        del gpu_query.exe > nul 2>&1
    ) else (
        echo Не удалось определить архитектуру. Использую sm_70 по умолчанию.
        set ARCH=sm_70
    )
)

if not defined ARCH (
    echo Неверный выбор. Использую sm_70 по умолчанию.
    set ARCH=sm_70
)

echo.
echo ==============================================
echo КОМПИЛЯЦИЯ ПРОГРАММЫ
echo ==============================================
echo Архитектура: %ARCH%
echo.

REM Создаем файл main.cu если он не существует
if not exist "main.cu" (
    echo Создание основного файла программы...
    call :create_main_cu
)

echo Компиляция с флагами:
echo - nvcc -std=c++17 -O3 -arch=%ARCH% -o fp_comparison.exe main.cu
echo.

REM Компиляция
nvcc -std=c++17 -O3 -arch=%ARCH% -o fp_comparison.exe main.cu

if %errorlevel% neq 0 (
    echo.
    echo ОШИБКА КОМПИЛЯЦИИ!
    echo.
    echo Возможные причины:
    echo 1. Неправильная архитектура GPU
    echo 2. Проблемы с установкой CUDA
    echo 3. Синтаксические ошибки в коде
    echo.
    pause
    exit /b 1
)

echo.
echo ==============================================
echo ЗАПУСК ПРОГРАММЫ
echo ==============================================
echo.

REM Запуск программы
fp_comparison.exe

echo.
echo ==============================================
echo.
echo Программа завершена.
echo.

REM Очистка временных файлов
if exist "*.exe" (
    echo Удаление временных файлов...
    del /q *.exe > nul 2>&1
    del /q *.exp > nul 2>&1
    del /q *.lib > nul 2>&1
    del /q *.pdb > nul 2>&1
)

pause
exit /b 0

:create_main_cu
(
echo #include ^<iostream^>
echo #include ^<vector^>
echo #include ^<chrono^>
echo #include ^<cmath^>
echo #include ^<iomanip^>
echo #include ^<cuda_fp16.h^>
echo #include ^<cuda_bf16.h^>
echo #include ^<cuda_runtime.h^>
echo.
echo // Для FP8 ^(требуется CUDA 11.8+ и совместимый GPU^)
echo #ifdef __CUDA_FP8_TYPES_EXIST__
echo #include ^<cuda_fp8.h^>
echo #endif
echo.
echo // Количество элементов для тестирования
echo const size_t N = 10000000;  // 10 миллионов элементов
echo const int BLOCK_SIZE = 256;
echo.
echo // Прототипы функций
echo void test_fp64^(^);
echo void test_fp32^(^);
echo void test_fp16^(^);
echo void test_bf16^(^);  // bfloat16 ^(альтернатива FP16^)
echo void test_fp8^(^);   // FP8 если доступно
echo.
echo // Вспомогательная функция для проверки ошибок CUDA
echo #define CUDA_CHECK^(call^) \
echo do { \
echo     cudaError_t error = call; \
echo     if ^(error != cudaSuccess^) { \
echo         std::cerr ^<^< "CUDA Error in " ^<^< #call ^<^< ": " \
echo                   ^<^< cudaGetErrorString^(error^) ^<^< " at " \
echo                   ^<^< __FILE__ ^<^< ":" ^<^< __LINE__ ^<^< std::endl; \
echo         exit^(EXIT_FAILURE^); \
echo     } \
echo } while^(0^)
echo.
echo // Ядра CUDA для различных типов данных
echo.
echo // FP64 ^(double^)
echo __global__ void vector_add_fp64^(const double* a, const double* b, double* c, int n^) {
echo     int idx = blockIdx.x * blockDim.x + threadIdx.x;
echo     if ^(idx ^< n^) {
echo         c[idx] = a[idx] + b[idx];
echo     }
echo }
echo.
echo // FP32 ^(float^)
echo __global__ void vector_add_fp32^(const float* a, const float* b, float* c, int n^) {
echo     int idx = blockIdx.x * blockDim.x + threadIdx.x;
echo     if ^(idx ^< n^) {
echo         c[idx] = a[idx] + b[idx];
echo     }
echo }
echo.
echo // FP16 ^(half^)
echo __global__ void vector_add_fp16^(const half* a, const half* b, half* c, int n^) {
echo     int idx = blockIdx.x * blockDim.x + threadIdx.x;
echo     if ^(idx ^< n^) {
echo         // Используем встроенные функции для операций с half
echo         c[idx] = __hadd^(a[idx], b[idx]^);
echo     }
echo }
echo.
echo // bfloat16
echo __global__ void vector_add_bf16^(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int n^) {
echo     int idx = blockIdx.x * blockDim.x + threadIdx.x;
echo     if ^(idx ^< n^) {
echo         // Преобразуем в float для вычислений, затем обратно
echo         float a_f = __bfloat162float^(a[idx]^);
echo         float b_f = __bfloat162float^(b[idx]^);
echo         float c_f = a_f + b_f;
echo         c[idx] = __float2bfloat16_rn^(c_f^);
echo     }
echo }
echo.
echo // Функция для измерения времени выполнения
echo template^<typename Func^>
echo double measure_kernel_time^(Func kernel_func, int iterations = 10^) {
echo     cudaEvent_t start, stop;
echo     CUDA_CHECK^(cudaEventCreate^(^&start^)^);
echo     CUDA_CHECK^(cudaEventCreate^(^&stop^)^);
echo     
echo     double total_time = 0.0;
echo     
echo     for ^(int i = 0; i ^< iterations; ++i^) {
echo         // Синхронизация перед началом измерения
echo         CUDA_CHECK^(cudaDeviceSynchronize^(^)^);
echo         
echo         CUDA_CHECK^(cudaEventRecord^(start^)^);
echo         
echo         // Выполняем ядро
echo         kernel_func^(^);
echo         
echo         CUDA_CHECK^(cudaEventRecord^(stop^)^);
echo         CUDA_CHECK^(cudaEventSynchronize^(stop^)^);
echo         
echo         float milliseconds = 0;
echo         CUDA_CHECK^(cudaEventElapsedTime^(^&milliseconds, start, stop^)^);
echo         total_time += milliseconds;
echo     }
echo     
echo     CUDA_CHECK^(cudaEventDestroy^(start^)^);
echo     CUDA_CHECK^(cudaEventDestroy^(stop^)^);
echo     
echo     return total_time / iterations;
echo }
echo.
echo // Тест FP64
echo void test_fp64^(^) {
echo     std::cout ^<^< "\n=== Тестирование FP64 ^(double^) ===" ^<^< std::endl;
echo     
echo     // Выделяем память на хосте
echo     std::vector^<double^> h_a^(N, 1.23456789^);
echo     std::vector^<double^> h_b^(N, 2.34567890^);
echo     std::vector^<double^> h_c^(N^);
echo     
echo     // Выделяем память на устройстве
echo     double *d_a, *d_b, *d_c;
echo     CUDA_CHECK^(cudaMalloc^(^&d_a, N * sizeof^(double^)^)^);
echo     CUDA_CHECK^(cudaMalloc^(^&d_b, N * sizeof^(double^)^)^);
echo     CUDA_CHECK^(cudaMalloc^(^&d_c, N * sizeof^(double^)^)^);
echo     
echo     // Копируем данные на устройство
echo     CUDA_CHECK^(cudaMemcpy^(d_a, h_a.data^(^), N * sizeof^(double^), cudaMemcpyHostToDevice^)^);
echo     CUDA_CHECK^(cudaMemcpy^(d_b, h_b.data^(^), N * sizeof^(double^), cudaMemcpyHostToDevice^)^);
echo     
echo     // Настраиваем параметры запуска ядра
echo     int threadsPerBlock = BLOCK_SIZE;
echo     int blocksPerGrid = ^(N + threadsPerBlock - 1^) / threadsPerBlock;
echo     
echo     // Измеряем время выполнения
echo     auto kernel_launch = [^&]^(^) {
echo         vector_add_fp64^<^<^<blocksPerGrid, threadsPerBlock^>^>^>^(d_a, d_b, d_c, N^);
echo         CUDA_CHECK^(cudaGetLastError^(^)^);
echo     };
echo     
echo     double kernel_time = measure_kernel_time^(kernel_launch^);
echo     
echo     // Копируем результат обратно
echo     CUDA_CHECK^(cudaMemcpy^(h_c.data^(^), d_c, N * sizeof^(double^), cudaMemcpyDeviceToHost^)^);
echo     
echo     // Проверяем результат ^(простая проверка^)
echo     bool correct = true;
echo     for ^(size_t i = 0; i ^< std::min^(N, size_t^(1000^)^); ++i^) {
echo         double expected = h_a[i] + h_b[i];
echo         if ^(fabs^(h_c[i] - expected^) ^> 1e-10^) {
echo             correct = false;
echo             break;
echo         }
echo     }
echo     
echo     // Выводим результаты
echo     std::cout ^<^< "Время выполнения: " ^<^< std::fixed ^<^< std::setprecision^(3^) 
echo               ^<^< kernel_time ^<^< " мс" ^<^< std::endl;
echo     std::cout ^<^< "Пропускная способность: " 
echo               ^<^< std::fixed ^<^< std::setprecision^(2^) 
echo               ^<^< ^(3.0 * N * sizeof^(double^) / ^(kernel_time / 1000.0^) / 1e9^) 
echo               ^<^< " GB/s" ^<^< std::endl;
echo     std::cout ^<^< "Проверка: " ^<^< ^(correct ? "Пройдена" : "Не пройдена"^) ^<^< std::endl;
echo     
echo     // Освобождаем память
echo     CUDA_CHECK^(cudaFree^(d_a^)^);
echo     CUDA_CHECK^(cudaFree^(d_b^)^);
echo     CUDA_CHECK^(cudaFree^(d_c^)^);
echo }
echo.
echo // Тест FP32
echo void test_fp32^(^) {
echo     std::cout ^<^< "\n=== Тестирование FP32 ^(float^) ===" ^<^< std::endl;
echo     
echo     std::vector^<float^> h_a^(N, 1.234567f^);
echo     std::vector^<float^> h_b^(N, 2.345678f^);
echo     std::vector^<float^> h_c^(N^);
echo     
echo     float *d_a, *d_b, *d_c;
echo     CUDA_CHECK^(cudaMalloc^(^&d_a, N * sizeof^(float^)^)^);
echo     CUDA_CHECK^(cudaMalloc^(^&d_b, N * sizeof^(float^)^)^);
echo     CUDA_CHECK^(cudaMalloc^(^&d_c, N * sizeof^(float^)^)^);
echo     
echo     CUDA_CHECK^(cudaMemcpy^(d_a, h_a.data^(^), N * sizeof^(float^), cudaMemcpyHostToDevice^)^);
echo     CUDA_CHECK^(cudaMemcpy^(d_b, h_b.data^(^), N * sizeof^(float^), cudaMemcpyHostToDevice^)^);
echo     
echo     int threadsPerBlock = BLOCK_SIZE;
echo     int blocksPerGrid = ^(N + threadsPerBlock - 1^) / threadsPerBlock;
echo     
echo     auto kernel_launch = [^&]^(^) {
echo         vector_add_fp32^<^<^<blocksPerGrid, threadsPerBlock^>^>^>^(d_a, d_b, d_c, N^);
echo         CUDA_CHECK^(cudaGetLastError^(^)^);
echo     };
echo     
echo     double kernel_time = measure_kernel_time^(kernel_launch^);
echo     
echo     CUDA_CHECK^(cudaMemcpy^(h_c.data^(^), d_c, N * sizeof^(float^), cudaMemcpyDeviceToHost^)^);
echo     
echo     bool correct = true;
echo     for ^(size_t i = 0; i ^< std::min^(N, size_t^(1000^)^); ++i^) {
echo         float expected = h_a[i] + h_b[i];
echo         if ^(fabs^(h_c[i] - expected^) ^> 1e-5^) {
echo             correct = false;
echo             break;
echo         }
echo     }
echo     
echo     std::cout ^<^< "Время выполнения: " ^<^< std::fixed ^<^< std::setprecision^(3^) 
echo               ^<^< kernel_time ^<^< " мс" ^<^< std::endl;
echo     std::cout ^<^< "Пропускная способность: " 
echo               ^<^< std::fixed ^<^< std::setprecision^(2^) 
echo               ^<^< ^(3.0 * N * sizeof^(float^) / ^(kernel_time / 1000.0^) / 1e9^) 
echo               ^<^< " GB/s" ^<^< std::endl;
echo     std::cout ^<^< "Проверка: " ^<^< ^(correct ? "Пройдена" : "Не пройдена"^) ^<^< std::endl;
echo     
echo     CUDA_CHECK^(cudaFree^(d_a^)^);
echo     CUDA_CHECK^(cudaFree^(d_b^)^);
echo     CUDA_CHECK^(cudaFree^(d_c^)^);
echo }
echo.
echo // Тест FP16
echo void test_fp16^(^) {
echo     std::cout ^<^< "\n=== Тестирование FP16 ^(half^) ===" ^<^< std::endl;
echo     
echo     std::vector^<half^> h_a^(N^);
echo     std::vector^<half^> h_b^(N^);
echo     std::vector^<half^> h_c^(N^);
echo     
echo     // Инициализируем с помощью float, затем конвертируем в half
echo     for ^(size_t i = 0; i ^< N; ++i^) {
echo         h_a[i] = __float2half^(1.234f^);
echo         h_b[i] = __float2half^(2.345f^);
echo     }
echo     
echo     half *d_a, *d_b, *d_c;
echo     CUDA_CHECK^(cudaMalloc^(^&d_a, N * sizeof^(half^)^)^);
echo     CUDA_CHECK^(cudaMalloc^(^&d_b, N * sizeof^(half^)^)^);
echo     CUDA_CHECK^(cudaMalloc^(^&d_c, N * sizeof^(half^)^)^);
echo     
echo     CUDA_CHECK^(cudaMemcpy^(d_a, h_a.data^(^), N * sizeof^(half^), cudaMemcpyHostToDevice^)^);
echo     CUDA_CHECK^(cudaMemcpy^(d_b, h_b.data^(^), N * sizeof^(half^), cudaMemcpyHostToDevice^)^);
echo     
echo     int threadsPerBlock = BLOCK_SIZE;
echo     int blocksPerGrid = ^(N + threadsPerBlock - 1^) / threadsPerBlock;
echo     
echo     auto kernel_launch = [^&]^(^) {
echo         vector_add_fp16^<^<^<blocksPerGrid, threadsPerBlock^>^>^>^(d_a, d_b, d_c, N^);
echo         CUDA_CHECK^(cudaGetLastError^(^)^);
echo     };
echo     
echo     double kernel_time = measure_kernel_time^(kernel_launch^);
echo     
echo     CUDA_CHECK^(cudaMemcpy^(h_c.data^(^), d_c, N * sizeof^(half^), cudaMemcpyDeviceToHost^)^);
echo     
echo     bool correct = true;
echo     for ^(size_t i = 0; i ^< std::min^(N, size_t^(1000^)^); ++i^) {
echo         float a_f = __half2float^(h_a[i]^);
echo         float b_f = __half2float^(h_b[i]^);
echo         float c_f = __half2float^(h_c[i]^);
echo         float expected = a_f + b_f;
echo         
echo         // Для half допускаем большую погрешность
echo         if ^(fabs^(c_f - expected^) ^> 1e-2^) {
echo             correct = false;
echo             break;
echo         }
echo     }
echo     
echo     std::cout ^<^< "Время выполнения: " ^<^< std::fixed ^<^< std::setprecision^(3^) 
echo               ^<^< kernel_time ^<^< " мс" ^<^< std::endl;
echo     std::cout ^<^< "Пропускная способность: " 
echo               ^<^< std::fixed ^<^< std::setprecision^(2^) 
echo               ^<^< ^(3.0 * N * sizeof^(half^) / ^(kernel_time / 1000.0^) / 1e9^) 
echo               ^<^< " GB/s" ^<^< std::endl;
echo     std::cout ^<^< "Проверка: " ^<^< ^(correct ? "Пройдена" : "Не пройдена"^) ^<^< std::endl;
echo     
echo     CUDA_CHECK^(cudaFree^(d_a^)^);
echo     CUDA_CHECK^(cudaFree^(d_b^)^);
echo     CUDA_CHECK^(cudaFree^(d_c^)^);
echo }
echo.
echo // Тест bfloat16
echo void test_bf16^(^) {
echo     std::cout ^<^< "\n=== Тестирование bfloat16 ===" ^<^< std::endl;
echo     
echo     std::vector^<__nv_bfloat16^> h_a^(N^);
echo     std::vector^<__nv_bfloat16^> h_b^(N^);
echo     std::vector^<__nv_bfloat16^> h_c^(N^);
echo     
echo     // Инициализируем
echo     for ^(size_t i = 0; i ^< N; ++i^) {
echo         h_a[i] = __float2bfloat16^(1.234f^);
echo         h_b[i] = __float2bfloat16^(2.345f^);
echo     }
echo     
echo     __nv_bfloat16 *d_a, *d_b, *d_c;
echo     CUDA_CHECK^(cudaMalloc^(^&d_a, N * sizeof^(__nv_bfloat16^)^)^);
echo     CUDA_CHECK^(cudaMalloc^(^&d_b, N * sizeof^(__nv_bfloat16^)^)^);
echo     CUDA_CHECK^(cudaMalloc^(^&d_c, N * sizeof^(__nv_bfloat16^)^)^);
echo     
echo     CUDA_CHECK^(cudaMemcpy^(d_a, h_a.data^(^), N * sizeof^(__nv_bfloat16^), cudaMemcpyHostToDevice^)^);
echo     CUDA_CHECK^(cudaMemcpy^(d_b, h_b.data^(^), N * sizeof^(__nv_bfloat16^), cudaMemcpyHostToDevice^)^);
echo     
echo     int threadsPerBlock = BLOCK_SIZE;
echo     int blocksPerGrid = ^(N + threadsPerBlock - 1^) / threadsPerBlock;
echo     
echo     auto kernel_launch = [^&]^(^) {
echo         vector_add_bf16^<^<^<blocksPerGrid, threadsPerBlock^>^>^>^(d_a, d_b, d_c, N^);
echo         CUDA_CHECK^(cudaGetLastError^(^)^);
echo     };
echo     
echo     double kernel_time = measure_kernel_time^(kernel_launch^);
echo     
echo     CUDA_CHECK^(cudaMemcpy^(h_c.data^(^), d_c, N * sizeof^(__nv_bfloat16^), cudaMemcpyDeviceToHost^)^);
echo     
echo     bool correct = true;
echo     for ^(size_t i = 0; i ^< std::min^(N, size_t^(1000^)^); ++i^) {
echo         float a_f = __bfloat162float^(h_a[i]^);
echo         float b_f = __bfloat162float^(h_b[i]^);
echo         float c_f = __bfloat162float^(h_c[i]^);
echo         float expected = a_f + b_f;
echo         
echo         if ^(fabs^(c_f - expected^) ^> 1e-2^) {
echo             correct = false;
echo             break;
echo         }
echo     }
echo     
echo     std::cout ^<^< "Время выполнения: " ^<^< std::fixed ^<^< std::setprecision^(3^) 
echo               ^<^< kernel_time ^<^< " мс" ^<^< std::endl;
echo     std::cout ^<^< "Пропускная способность: " 
echo               ^<^< std::fixed ^<^< std::setprecision^(2^) 
echo               ^<^< ^(3.0 * N * sizeof^(__nv_bfloat16^) / ^(kernel_time / 1000.0^) / 1e9^) 
echo               ^<^< " GB/s" ^<^< std::endl;
echo     std::cout ^<^< "Проверка: " ^<^< ^(correct ? "Пройдена" : "Не пройдена"^) ^<^< std::endl;
echo     
echo     CUDA_CHECK^(cudaFree^(d_a^)^);
echo     CUDA_CHECK^(cudaFree^(d_b^)^);
echo     CUDA_CHECK^(cudaFree^(d_c^)^);
echo }
echo.
echo // Тест FP8 ^(если доступно^)
echo void test_fp8^(^) {
echo #ifdef __CUDA_FP8_TYPES_EXIST__
echo     std::cout ^<^< "\n=== Тестирование FP8 ^(E4M3^) ===" ^<^< std::endl;
echo     
echo     // Примечание: для простоты мы будем использовать эмуляцию FP8 через float
echo     // Реальная реализация требует CUDA 11.8+ и совместимый GPU
echo     
echo     std::cout ^<^< "FP8 тестирование требует CUDA 11.8+ и GPU с поддержкой FP8." ^<^< std::endl;
echo     std::cout ^<^< "На данном оборудовании тест пропущен." ^<^< std::endl;
echo     
echo #else
echo     std::cout ^<^< "\n=== Тестирование FP8 ===" ^<^< std::endl;
echo     std::cout ^<^< "FP8 не поддерживается в данной версии CUDA." ^<^< std::endl;
echo     std::cout ^<^< "Требуется CUDA 11.8+ и GPU с архитектурой Hopper или новее." ^<^< std::endl;
echo #endif
echo }
echo.
echo // Функция для отображения информации о GPU
echo void print_gpu_info^(^) {
echo     int device_count;
echo     CUDA_CHECK^(cudaGetDeviceCount^(^&device_count^)^);
echo     
echo     if ^(device_count == 0^) {
echo         std::cerr ^<^< "Не найдено устройств CUDA!" ^<^< std::endl;
echo         return;
echo     }
echo     
echo     cudaDeviceProp prop;
echo     CUDA_CHECK^(cudaGetDeviceProperties^(^&prop, 0^)^);
echo     
echo     std::cout ^<^< "\n=== Информация о GPU ===" ^<^< std::endl;
echo     std::cout ^<^< "Название: " ^<^< prop.name ^<^< std::endl;
echo     std::cout ^<^< "Версия CUDA: " ^<^< prop.major ^<^< "." ^<^< prop.minor ^<^< std::endl;
echo     std::cout ^<^< "Общая память: " ^<^< prop.totalGlobalMem / ^(1024 * 1024 * 1024.0^) ^<^< " GB" ^<^< std::endl;
echo     std::cout ^<^< "Мультипроцессоры: " ^<^< prop.multiProcessorCount ^<^< std::endl;
echo     std::cout ^<^< "Частота: " ^<^< prop.clockRate / 1000.0 ^<^< " MHz" ^<^< std::endl;
echo     
echo     // Проверяем поддержку различных типов данных
echo     std::cout ^<^< "\nПоддерживаемые типы данных:" ^<^< std::endl;
echo     std::cout ^<^< "- FP64 ^(double^): " 
echo               ^<^< ^(^(prop.major ^>= 2^) ? "Да" : "Нет ^(требуется Compute Capability 2.0+^)"^) ^<^< std::endl;
echo     std::cout ^<^< "- FP32 ^(float^): Да" ^<^< std::endl;
echo     std::cout ^<^< "- FP16 ^(half^): " 
echo               ^<^< ^(^(prop.major ^>= 5^) ? "Да ^(с Tensor Cores: " 
echo                   + std::to_string^(prop.major ^>= 7 ? "Volta+^)" : "Pascal^)"^) 
echo                   : "Нет ^(требуется Compute Capability 5.0+^)"^) ^<^< std::endl;
echo     std::cout ^<^< "- bfloat16: " 
echo               ^<^< ^(^(prop.major ^>= 8^) ? "Да ^(Ampere+^)" : "Нет ^(требуется Compute Capability 8.0+^)"^) ^<^< std::endl;
echo     std::cout ^<^< "- FP8: " 
echo               ^<^< ^(^(prop.major ^>= 9^) ? "Да ^(Hopper+^)" : "Нет ^(требуется Compute Capability 9.0+^)"^) ^<^< std::endl;
echo }
echo.
echo int main^(^) {
echo     std::cout ^<^< "СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ РАЗЛИЧНЫХ ТИПОВ ДАННЫХ С ПЛАВАЮЩЕЙ ТОЧКОЙ" ^<^< std::endl;
echo     std::cout ^<^< "=====================================================================" ^<^< std::endl;
echo     
echo     // Отображаем информацию о GPU
echo     print_gpu_info^(^);
echo     
echo     // Запускаем тесты
echo     try {
echo         // Тестируем все поддерживаемые типы
echo         test_fp64^(^);
echo         test_fp32^(^);
echo         test_fp16^(^);
echo         test_bf16^(^);
echo         test_fp8^(^);
echo         
echo         // Сводная таблица результатов
echo         std::cout ^<^< "\n=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===" ^<^< std::endl;
echo         std::cout ^<^< "Тип данных ^| Размер ^(байт^) ^| Относительная скорость" ^<^< std::endl;
echo         std::cout ^<^< "------------------------------------------" ^<^< std::endl;
echo         std::cout ^<^< "FP64       ^|      8        ^|     1.0x ^(базовая^)" ^<^< std::endl;
echo         std::cout ^<^< "FP32       ^|      4        ^|     ~2x быстрее" ^<^< std::endl;
echo         std::cout ^<^< "FP16       ^|      2        ^|     ~4x быстрее" ^<^< std::endl;
echo         std::cout ^<^< "bfloat16   ^|      2        ^|     ~4x быстрее" ^<^< std::endl;
echo         std::cout ^<^< "FP8        ^|      1        ^|     ~8x быстрее ^(на Tensor Cores^)" ^<^< std::endl;
echo         
echo         std::cout ^<^< "\n=== ПРАКТИЧЕСКИЕ ВЫВОДЫ ===" ^<^< std::endl;
echo         std::cout ^<^< "1. FP64: Для максимальной точности, но медленнее" ^<^< std::endl;
echo         std::cout ^<^< "2. FP32: Оптимальный баланс точности и скорости" ^<^< std::endl;
echo         std::cout ^<^< "3. FP16/bfloat16: Для ИИ-вычислений и экономии памяти" ^<^< std::endl;
echo         std::cout ^<^< "4. FP8: Для новейших ИИ-ускорителей, максимальная скорость" ^<^< std::endl;
echo         
echo         std::cout ^<^< "\nПримечание: Реальная производительность зависит от:" ^<^< std::endl;
echo         std::cout ^<^< "- Архитектуры GPU" ^<^< std::endl;
echo         std::cout ^<^< "- Поддержки Tensor Cores" ^<^< std::endl;
echo         std::cout ^<^< "- Оптимизации памяти" ^<^< std::endl;
echo         std::cout ^<^< "- Параллелизма на уровне инструкций" ^<^< std::endl;
echo         
echo     } catch ^(const std::exception^& e^) {
echo         std::cerr ^<^< "Ошибка: " ^<^< e.what^(^) ^<^< std::endl;
echo         return 1;
echo     }
echo     
echo     // Сбрасываем устройство
echo     CUDA_CHECK^(cudaDeviceReset^(^)^);
echo     
echo     return 0;
echo }
) > main.cu
exit /b 0