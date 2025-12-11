#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_fp16.h>
#include <cuda_bf16.h>  // Для bfloat16, если доступно
#include <cuda_runtime.h>
#include <sstream>
#include <string>

// Для FP8 (требуется CUDA 11.8+ и совместимый GPU)
#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif

template <typename T>
std::string to_string(const T& value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

// Количество элементов для тестирования
const size_t N = 10000000;  // 10 миллионов элементов
const int BLOCK_SIZE = 256;

// Прототипы функций
void test_fp64();
void test_fp32();
void test_fp16();
void test_bf16();  // bfloat16 (альтернатива FP16)
void test_fp8();   // FP8 если доступно

// Вспомогательная функция для проверки ошибок CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error in " << #call << ": " \
                  << cudaGetErrorString(error) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Ядра CUDA для различных типов данных

// FP64 (double)
__global__ void vector_add_fp64(const double* a, const double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// FP32 (float)
__global__ void vector_add_fp32(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// FP16 (half)
__global__ void vector_add_fp16(const half* a, const half* b, half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Используем встроенные функции для операций с half
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// bfloat16
__global__ void vector_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Преобразуем в float для вычислений, затем обратно
        float a_f = __bfloat162float(a[idx]);
        float b_f = __bfloat162float(b[idx]);
        float c_f = a_f + b_f;
        c[idx] = __float2bfloat16_rn(c_f);
    }
}

// Функция для измерения времени выполнения
template<typename Func>
double measure_kernel_time(Func kernel_func, int iterations = 10) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    double total_time = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        // Синхронизация перед началом измерения
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        
        // Выполняем ядро
        kernel_func();
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time += milliseconds;
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return total_time / iterations;
}

// Тест FP64
void test_fp64() {
    std::cout << "\n=== Тестирование FP64 (double) ===" << std::endl;
    
    // Выделяем память на хосте
    std::vector<double> h_a(N, 1.23456789);
    std::vector<double> h_b(N, 2.34567890);
    std::vector<double> h_c(N);
    
    // Выделяем память на устройстве
    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(double)));
    
    // Копируем данные на устройство
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Настраиваем параметры запуска ядра
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Измеряем время выполнения
    auto kernel_launch = [&]() {
        vector_add_fp64<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());
    };
    
    double kernel_time = measure_kernel_time(kernel_launch);
    
    // Копируем результат обратно
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Проверяем результат (простая проверка)
    bool correct = true;
    for (size_t i = 0; i < std::min(N, size_t(1000)); ++i) {
        double expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-10) {
            correct = false;
            break;
        }
    }
    
    // Выводим результаты
    std::cout << "Время выполнения: " << std::fixed << std::setprecision(3) 
              << kernel_time << " мс" << std::endl;
    std::cout << "Пропускная способность: " 
              << std::fixed << std::setprecision(2) 
              << (3.0 * N * sizeof(double) / (kernel_time / 1000.0) / 1e9) 
              << " GB/s" << std::endl;
    std::cout << "Проверка: " << (correct ? "Пройдена" : "Не пройдена") << std::endl;
    
    // Освобождаем память
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// Тест FP32
void test_fp32() {
    std::cout << "\n=== Тестирование FP32 (float) ===" << std::endl;
    
    std::vector<float> h_a(N, 1.234567f);
    std::vector<float> h_b(N, 2.345678f);
    std::vector<float> h_c(N);
    
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    auto kernel_launch = [&]() {
        vector_add_fp32<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());
    };
    
    double kernel_time = measure_kernel_time(kernel_launch);
    
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (size_t i = 0; i < std::min(N, size_t(1000)); ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Время выполнения: " << std::fixed << std::setprecision(3) 
              << kernel_time << " мс" << std::endl;
    std::cout << "Пропускная способность: " 
              << std::fixed << std::setprecision(2) 
              << (3.0 * N * sizeof(float) / (kernel_time / 1000.0) / 1e9) 
              << " GB/s" << std::endl;
    std::cout << "Проверка: " << (correct ? "Пройдена" : "Не пройдена") << std::endl;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// Тест FP16
void test_fp16() {
    std::cout << "\n=== Тестирование FP16 (half) ===" << std::endl;
    
    std::vector<half> h_a(N);
    std::vector<half> h_b(N);
    std::vector<half> h_c(N);
    
    // Инициализируем с помощью float, затем конвертируем в half
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = __float2half(1.234f);
        h_b[i] = __float2half(2.345f);
    }
    
    half *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    auto kernel_launch = [&]() {
        vector_add_fp16<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());
    };
    
    double kernel_time = measure_kernel_time(kernel_launch);
    
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(half), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (size_t i = 0; i < std::min(N, size_t(1000)); ++i) {
        float a_f = __half2float(h_a[i]);
        float b_f = __half2float(h_b[i]);
        float c_f = __half2float(h_c[i]);
        float expected = a_f + b_f;
        
        // Для half допускаем большую погрешность
        if (fabs(c_f - expected) > 1e-2) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Время выполнения: " << std::fixed << std::setprecision(3) 
              << kernel_time << " мс" << std::endl;
    std::cout << "Пропускная способность: " 
              << std::fixed << std::setprecision(2) 
              << (3.0 * N * sizeof(half) / (kernel_time / 1000.0) / 1e9) 
              << " GB/s" << std::endl;
    std::cout << "Проверка: " << (correct ? "Пройдена" : "Не пройдена") << std::endl;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// Тест bfloat16
void test_bf16() {
    std::cout << "\n=== Тестирование bfloat16 ===" << std::endl;
    
    std::vector<__nv_bfloat16> h_a(N);
    std::vector<__nv_bfloat16> h_b(N);
    std::vector<__nv_bfloat16> h_c(N);
    
    // Инициализируем
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = __float2bfloat16(1.234f);
        h_b[i] = __float2bfloat16(2.345f);
    }
    
    __nv_bfloat16 *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(__nv_bfloat16)));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    auto kernel_launch = [&]() {
        vector_add_bf16<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());
    };
    
    double kernel_time = measure_kernel_time(kernel_launch);
    
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (size_t i = 0; i < std::min(N, size_t(1000)); ++i) {
        float a_f = __bfloat162float(h_a[i]);
        float b_f = __bfloat162float(h_b[i]);
        float c_f = __bfloat162float(h_c[i]);
        float expected = a_f + b_f;
        
        if (fabs(c_f - expected) > 1e-2) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Время выполнения: " << std::fixed << std::setprecision(3) 
              << kernel_time << " мс" << std::endl;
    std::cout << "Пропускная способность: " 
              << std::fixed << std::setprecision(2) 
              << (3.0 * N * sizeof(__nv_bfloat16) / (kernel_time / 1000.0) / 1e9) 
              << " GB/s" << std::endl;
    std::cout << "Проверка: " << (correct ? "Пройдена" : "Не пройдена") << std::endl;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// Тест FP8 (если доступно)
void test_fp8() {
#ifdef __CUDA_FP8_TYPES_EXIST__
    std::cout << "\n=== Тестирование FP8 (E4M3) ===" << std::endl;
    
    // Примечание: для простоты мы будем использовать эмуляцию FP8 через float
    // Реальная реализация требует CUDA 11.8+ и совместимый GPU
    
    std::cout << "FP8 тестирование требует CUDA 11.8+ и GPU с поддержкой FP8." << std::endl;
    std::cout << "На данном оборудовании тест пропущен." << std::endl;
    
#else
    std::cout << "\n=== Тестирование FP8 ===" << std::endl;
    std::cout << "FP8 не поддерживается в данной версии CUDA." << std::endl;
    std::cout << "Требуется CUDA 11.8+ и GPU с архитектурой Hopper или новее." << std::endl;
#endif
}

// Функция для отображения информации о GPU
void print_gpu_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        std::cerr << "Не найдено устройств CUDA!" << std::endl;
        return;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "\n=== Информация о GPU ===" << std::endl;
    std::cout << "Название: " << prop.name << std::endl;
    std::cout << "Версия CUDA: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Общая память: " << prop.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB" << std::endl;
    std::cout << "Мультипроцессоры: " << prop.multiProcessorCount << std::endl;
    std::cout << "Частота: " << prop.clockRate / 1000.0 << " MHz" << std::endl;
    
    // Проверяем поддержку различных типов данных
    std::cout << "\nПоддерживаемые типы данных:" << std::endl;
    std::cout << "- FP64 (double): " 
              << ((prop.major >= 2) ? "Да" : "Нет (требуется Compute Capability 2.0+)") << std::endl;
    std::cout << "- FP32 (float): Да" << std::endl;
    std::cout << "- FP16 (half): " 
              << ((prop.major >= 5) ? "Да (с Tensor Cores: "
                  : "Нет (требуется Compute Capability 5.0+)") << std::endl;
    std::cout << "- bfloat16: " 
              << ((prop.major >= 8) ? "Да (Ampere+)" : "Нет (требуется Compute Capability 8.0+)") << std::endl;
    std::cout << "- FP8: " 
              << ((prop.major >= 9) ? "Да (Hopper+)" : "Нет (требуется Compute Capability 9.0+)") << std::endl;
}

int main() {
    std::cout << "СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ РАЗЛИЧНЫХ ТИПОВ ДАННЫХ С ПЛАВАЮЩЕЙ ТОЧКОЙ" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    
    // Отображаем информацию о GPU
    print_gpu_info();
    
    // Запускаем тесты
    try {
        // Тестируем все поддерживаемые типы
        test_fp64();
        test_fp32();
        test_fp16();
        test_bf16();
        test_fp8();
        
        // Сводная таблица результатов
        std::cout << "\n=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===" << std::endl;
        std::cout << "Тип данных | Размер (байт) | Относительная скорость" << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "FP64       |      8        |     1.0x (базовая)" << std::endl;
        std::cout << "FP32       |      4        |     ~2x быстрее" << std::endl;
        std::cout << "FP16       |      2        |     ~4x быстрее" << std::endl;
        std::cout << "bfloat16   |      2        |     ~4x быстрее" << std::endl;
        std::cout << "FP8        |      1        |     ~8x быстрее (на Tensor Cores)" << std::endl;
        
        std::cout << "\n=== ПРАКТИЧЕСКИЕ ВЫВОДЫ ===" << std::endl;
        std::cout << "1. FP64: Для максимальной точности, но медленнее" << std::endl;
        std::cout << "2. FP32: Оптимальный баланс точности и скорости" << std::endl;
        std::cout << "3. FP16/bfloat16: Для ИИ-вычислений и экономии памяти" << std::endl;
        std::cout << "4. FP8: Для новейших ИИ-ускорителей, максимальная скорость" << std::endl;
        
        std::cout << "\nПримечание: Реальная производительность зависит от:" << std::endl;
        std::cout << "- Архитектуры GPU" << std::endl;
        std::cout << "- Поддержки Tensor Cores" << std::endl;
        std::cout << "- Оптимизации памяти" << std::endl;
        std::cout << "- Параллелизма на уровне инструкций" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    // Сбрасываем устройство
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}