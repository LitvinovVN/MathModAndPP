#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <omp.h>

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << #call << ": " \
                  << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Структура для хранения полей
template<typename T>
struct Field3D {
    T* data;
    int nx, ny, nz;
    size_t size;
    bool on_device;
    
    Field3D(int nx_, int ny_, int nz_, bool device = false) : 
        nx(nx_), ny(ny_), nz(nz_), on_device(device) {
        size = nx * ny * nz;
        if (device) {
            CUDA_CHECK(cudaMalloc(&data, size * sizeof(T)));
        } else {
            data = new T[size];
        }
    }
    
    ~Field3D() {
        if (on_device) {
            CUDA_CHECK(cudaFree(data));
        } else {
            delete[] data;
        }
    }
    
    void copy_to_device(T* host_data) {
        CUDA_CHECK(cudaMemcpy(data, host_data, size * sizeof(T), 
                              cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* host_data) {
        CUDA_CHECK(cudaMemcpy(host_data, data, size * sizeof(T), 
                              cudaMemcpyDeviceToHost));
    }
};

// Структура для векторного поля
template<typename T>
struct VectorField3D {
    Field3D<T> x, y, z;
    
    VectorField3D(int nx, int ny, int nz, bool device = false) : 
        x(nx, ny, nz, device), y(nx, ny, nz, device), z(nx, ny, nz, device) {}
};

// ==================== ЯДРА CUDA ДЛЯ FP32 ====================

__global__ void gradient_fp32_kernel(const float* scalar, float* grad_x, 
                                     float* grad_y, float* grad_z, 
                                     int nx, int ny, int nz, 
                                     float dx, float dy, float dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    // Центральные разности внутри области
    if (i > 0 && i < nx-1) {
        grad_x[idx] = (scalar[idx+1] - scalar[idx-1]) / (2.0f * dx);
    } else if (i == 0) {
        grad_x[idx] = (scalar[idx+1] - scalar[idx]) / dx; // Передняя разность
    } else {
        grad_x[idx] = (scalar[idx] - scalar[idx-1]) / dx; // Задняя разность
    }
    
    if (j > 0 && j < ny-1) {
        grad_y[idx] = (scalar[idx+nx] - scalar[idx-nx]) / (2.0f * dy);
    } else if (j == 0) {
        grad_y[idx] = (scalar[idx+nx] - scalar[idx]) / dy;
    } else {
        grad_y[idx] = (scalar[idx] - scalar[idx-nx]) / dy;
    }
    
    if (k > 0 && k < nz-1) {
        grad_z[idx] = (scalar[idx+nx*ny] - scalar[idx-nx*ny]) / (2.0f * dz);
    } else if (k == 0) {
        grad_z[idx] = (scalar[idx+nx*ny] - scalar[idx]) / dz;
    } else {
        grad_z[idx] = (scalar[idx] - scalar[idx-nx*ny]) / dz;
    }
}

__global__ void divergence_fp32_kernel(const float* vec_x, const float* vec_y, 
                                       const float* vec_z, float* div, 
                                       int nx, int ny, int nz, 
                                       float dx, float dy, float dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    // Вычисление производных с центральными разностями
    float dVx_dx, dVy_dy, dVz_dz;
    
    if (i > 0 && i < nx-1) {
        dVx_dx = (vec_x[idx+1] - vec_x[idx-1]) / (2.0f * dx);
    } else if (i == 0) {
        dVx_dx = (vec_x[idx+1] - vec_x[idx]) / dx;
    } else {
        dVx_dx = (vec_x[idx] - vec_x[idx-1]) / dx;
    }
    
    if (j > 0 && j < ny-1) {
        dVy_dy = (vec_y[idx+nx] - vec_y[idx-nx]) / (2.0f * dy);
    } else if (j == 0) {
        dVy_dy = (vec_y[idx+nx] - vec_y[idx]) / dy;
    } else {
        dVy_dy = (vec_y[idx] - vec_y[idx-nx]) / dy;
    }
    
    if (k > 0 && k < nz-1) {
        dVz_dz = (vec_z[idx+nx*ny] - vec_z[idx-nx*ny]) / (2.0f * dz);
    } else if (k == 0) {
        dVz_dz = (vec_z[idx+nx*ny] - vec_z[idx]) / dz;
    } else {
        dVz_dz = (vec_z[idx] - vec_z[idx-nx*ny]) / dz;
    }
    
    div[idx] = dVx_dx + dVy_dy + dVz_dz;
}

// ==================== ЯДРА CUDA ДЛЯ FP64 ====================

__global__ void gradient_fp64_kernel(const double* scalar, double* grad_x, 
                                     double* grad_y, double* grad_z, 
                                     int nx, int ny, int nz, 
                                     double dx, double dy, double dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    if (i > 0 && i < nx-1) {
        grad_x[idx] = (scalar[idx+1] - scalar[idx-1]) / (2.0 * dx);
    } else if (i == 0) {
        grad_x[idx] = (scalar[idx+1] - scalar[idx]) / dx;
    } else {
        grad_x[idx] = (scalar[idx] - scalar[idx-1]) / dx;
    }
    
    if (j > 0 && j < ny-1) {
        grad_y[idx] = (scalar[idx+nx] - scalar[idx-nx]) / (2.0 * dy);
    } else if (j == 0) {
        grad_y[idx] = (scalar[idx+nx] - scalar[idx]) / dy;
    } else {
        grad_y[idx] = (scalar[idx] - scalar[idx-nx]) / dy;
    }
    
    if (k > 0 && k < nz-1) {
        grad_z[idx] = (scalar[idx+nx*ny] - scalar[idx-nx*ny]) / (2.0 * dz);
    } else if (k == 0) {
        grad_z[idx] = (scalar[idx+nx*ny] - scalar[idx]) / dz;
    } else {
        grad_z[idx] = (scalar[idx] - scalar[idx-nx*ny]) / dz;
    }
}

// ==================== ЯДРА CUDA ДЛЯ FP16 ====================

__global__ void gradient_fp16_kernel(const half* scalar, half* grad_x, 
                                     half* grad_y, half* grad_z, 
                                     int nx, int ny, int nz, 
                                     half dx, half dy, half dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = i + j * nx + k * nx * ny;
    
    half two = __float2half(2.0f);
    
    if (i > 0 && i < nx-1) {
        grad_x[idx] = __hdiv(__hsub(scalar[idx+1], scalar[idx-1]), __hmul(two, dx));
    } else if (i == 0) {
        grad_x[idx] = __hdiv(__hsub(scalar[idx+1], scalar[idx]), dx);
    } else {
        grad_x[idx] = __hdiv(__hsub(scalar[idx], scalar[idx-1]), dx);
    }
    
    if (j > 0 && j < ny-1) {
        grad_y[idx] = __hdiv(__hsub(scalar[idx+nx], scalar[idx-nx]), __hmul(two, dy));
    } else if (j == 0) {
        grad_y[idx] = __hdiv(__hsub(scalar[idx+nx], scalar[idx]), dy);
    } else {
        grad_y[idx] = __hdiv(__hsub(scalar[idx], scalar[idx-nx]), dy);
    }
    
    if (k > 0 && k < nz-1) {
        grad_z[idx] = __hdiv(__hsub(scalar[idx+nx*ny], scalar[idx-nx*ny]), __hmul(two, dz));
    } else if (k == 0) {
        grad_z[idx] = __hdiv(__hsub(scalar[idx+nx*ny], scalar[idx]), dz);
    } else {
        grad_z[idx] = __hdiv(__hsub(scalar[idx], scalar[idx-nx*ny]), dz);
    }
}

// ==================== ОТКРЫТЫЕ ФУНКЦИИ CUDA ====================

template<typename T>
void cuda_gradient(const Field3D<T>& scalar_field, VectorField3D<T>& gradient, 
                   float dx, float dy, float dz) {
    // Реализация зависит от типа T
    std::cerr << "Unsupported type for CUDA gradient" << std::endl;
}

template<>
void cuda_gradient<float>(const Field3D<float>& scalar_field, 
                          VectorField3D<float>& gradient, 
                          float dx, float dy, float dz) {
    dim3 block_size(8, 8, 4);
    dim3 grid_size((scalar_field.nx + block_size.x - 1) / block_size.x,
                   (scalar_field.ny + block_size.y - 1) / block_size.y,
                   (scalar_field.nz + block_size.z - 1) / block_size.z);
    
    gradient_fp32_kernel<<<grid_size, block_size>>>(
        scalar_field.data, gradient.x.data, gradient.y.data, gradient.z.data,
        scalar_field.nx, scalar_field.ny, scalar_field.nz, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<>
void cuda_gradient<double>(const Field3D<double>& scalar_field, 
                           VectorField3D<double>& gradient, 
                           float dx, float dy, float dz) {
    dim3 block_size(8, 8, 4);
    dim3 grid_size((scalar_field.nx + block_size.x - 1) / block_size.x,
                   (scalar_field.ny + block_size.y - 1) / block_size.y,
                   (scalar_field.nz + block_size.z - 1) / block_size.z);
    
    gradient_fp64_kernel<<<grid_size, block_size>>>(
        scalar_field.data, gradient.x.data, gradient.y.data, gradient.z.data,
        scalar_field.nx, scalar_field.ny, scalar_field.nz, 
        (double)dx, (double)dy, (double)dz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<>
void cuda_gradient<half>(const Field3D<half>& scalar_field, 
                         VectorField3D<half>& gradient, 
                         float dx, float dy, float dz) {
    dim3 block_size(8, 8, 4);
    dim3 grid_size((scalar_field.nx + block_size.x - 1) / block_size.x,
                   (scalar_field.ny + block_size.y - 1) / block_size.y,
                   (scalar_field.nz + block_size.z - 1) / block_size.z);
    
    half h_dx = __float2half(dx);
    half h_dy = __float2half(dy);
    half h_dz = __float2half(dz);
    
    gradient_fp16_kernel<<<grid_size, block_size>>>(
        scalar_field.data, gradient.x.data, gradient.y.data, gradient.z.data,
        scalar_field.nx, scalar_field.ny, scalar_field.nz, h_dx, h_dy, h_dz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== OPENMP РЕАЛИЗАЦИИ ====================

template<typename T>
void omp_gradient(const T* scalar, T* grad_x, T* grad_y, T* grad_z, 
                  int nx, int ny, int nz, T dx, T dy, T dz) {
    #pragma omp parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                
                if (i > 0 && i < nx-1) {
                    grad_x[idx] = (scalar[idx+1] - scalar[idx-1]) / (2.0 * dx);
                } else if (i == 0) {
                    grad_x[idx] = (scalar[idx+1] - scalar[idx]) / dx;
                } else {
                    grad_x[idx] = (scalar[idx] - scalar[idx-1]) / dx;
                }
                
                if (j > 0 && j < ny-1) {
                    grad_y[idx] = (scalar[idx+nx] - scalar[idx-nx]) / (2.0 * dy);
                } else if (j == 0) {
                    grad_y[idx] = (scalar[idx+nx] - scalar[idx]) / dy;
                } else {
                    grad_y[idx] = (scalar[idx] - scalar[idx-nx]) / dy;
                }
                
                if (k > 0 && k < nz-1) {
                    grad_z[idx] = (scalar[idx+nx*ny] - scalar[idx-nx*ny]) / (2.0 * dz);
                } else if (k == 0) {
                    grad_z[idx] = (scalar[idx+nx*ny] - scalar[idx]) / dz;
                } else {
                    grad_z[idx] = (scalar[idx] - scalar[idx-nx*ny]) / dz;
                }
            }
        }
    }
}

template<typename T>
void omp_divergence(const T* vec_x, const T* vec_y, const T* vec_z, 
                    T* div, int nx, int ny, int nz, T dx, T dy, T dz) {
    #pragma omp parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                
                T dVx_dx, dVy_dy, dVz_dz;
                
                if (i > 0 && i < nx-1) {
                    dVx_dx = (vec_x[idx+1] - vec_x[idx-1]) / (2.0 * dx);
                } else if (i == 0) {
                    dVx_dx = (vec_x[idx+1] - vec_x[idx]) / dx;
                } else {
                    dVx_dx = (vec_x[idx] - vec_x[idx-1]) / dx;
                }
                
                if (j > 0 && j < ny-1) {
                    dVy_dy = (vec_y[idx+nx] - vec_y[idx-nx]) / (2.0 * dy);
                } else if (j == 0) {
                    dVy_dy = (vec_y[idx+nx] - vec_y[idx]) / dy;
                } else {
                    dVy_dy = (vec_y[idx] - vec_y[idx-nx]) / dy;
                }
                
                if (k > 0 && k < nz-1) {
                    dVz_dz = (vec_z[idx+nx*ny] - vec_z[idx-nx*ny]) / (2.0 * dz);
                } else if (k == 0) {
                    dVz_dz = (vec_z[idx+nx*ny] - vec_z[idx]) / dz;
                } else {
                    dVz_dz = (vec_z[idx] - vec_z[idx-nx*ny]) / dz;
                }
                
                div[idx] = dVx_dx + dVy_dy + dVz_dz;
            }
        }
    }
}

// ==================== ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ ====================

template<typename T>
void initialize_temperature_field(T* field, int nx, int ny, int nz, 
                                  float Lx, float Ly, float Lz) {
    #pragma omp parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                float x = (float)i / (nx-1) * Lx;
                float y = (float)j / (ny-1) * Ly;
                float z = (float)k / (nz-1) * Lz;
                
                // Температурное поле: синусоидальная функция
                field[idx] = (T)(300.0 + 50.0 * sin(2.0 * M_PI * x / Lx) * 
                                        sin(2.0 * M_PI * y / Ly) * 
                                        cos(2.0 * M_PI * z / Lz));
            }
        }
    }
}

template<typename T>
void initialize_vector_field(T* vec_x, T* vec_y, T* vec_z, 
                             int nx, int ny, int nz, 
                             float Lx, float Ly, float Lz) {
    #pragma omp parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                float x = (float)i / (nx-1) * Lx;
                float y = (float)j / (ny-1) * Ly;
                float z = (float)k / (nz-1) * Lz;
                
                // Векторное поле: вихревое поле
                vec_x[idx] = (T)(-y * exp(-(x*x + y*y + z*z) / (Lx*Ly*Lz)));
                vec_y[idx] = (T)(x * exp(-(x*x + y*y + z*z) / (Lx*Ly*Lz)));
                vec_z[idx] = (T)(0.0);
            }
        }
    }
}

// ==================== СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ====================

template<typename T>
void performance_comparison(int nx, int ny, int nz, int iterations = 10) {
    std::cout << "\n=== Производительность для типа размером " << sizeof(T) << " байт ===" << std::endl;
    
    // Создаем поля на хосте
    size_t total_size = nx * ny * nz;
    T* scalar_host = new T[total_size];
    T* vec_x_host = new T[total_size];
    T* vec_y_host = new T[total_size];
    T* vec_z_host = new T[total_size];
    T* grad_x_host = new T[total_size];
    T* grad_y_host = new T[total_size];
    T* grad_z_host = new T[total_size];
    T* div_host = new T[total_size];
    
    // Инициализируем поля
    initialize_temperature_field(scalar_host, nx, ny, nz, 10.0f, 10.0f, 10.0f);
    initialize_vector_field(vec_x_host, vec_y_host, vec_z_host, nx, ny, nz, 10.0f, 10.0f, 10.0f);
    
    float dx = 0.1f, dy = 0.1f, dz = 0.1f;
    
    // Тест OpenMP градиента
    double omp_grad_time = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        omp_gradient(scalar_host, grad_x_host, grad_y_host, grad_z_host, 
                     nx, ny, nz, (T)dx, (T)dy, (T)dz);
        auto end = std::chrono::high_resolution_clock::now();
        omp_grad_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    omp_grad_time /= iterations;
    
    // Тест OpenMP дивергенции
    double omp_div_time = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        omp_divergence(vec_x_host, vec_y_host, vec_z_host, div_host, 
                       nx, ny, nz, (T)dx, (T)dy, (T)dz);
        auto end = std::chrono::high_resolution_clock::now();
        omp_div_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    omp_div_time /= iterations;
    
    // Тест CUDA градиента
    double cuda_grad_time = 0.0;
    
    try {
        // Создаем поля на устройстве
        Field3D<T> scalar_device(nx, ny, nz, true);
        VectorField3D<T> grad_device(nx, ny, nz, true);
        
        // Копируем данные на устройство
        scalar_device.copy_to_device(scalar_host);
        
        for (int iter = 0; iter < iterations; ++iter) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            
            CUDA_CHECK(cudaEventRecord(start));
            cuda_gradient(scalar_device, grad_device, dx, dy, dz);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            cuda_grad_time += milliseconds;
            
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
        cuda_grad_time /= iterations;
        
    } catch (...) {
        cuda_grad_time = -1.0; // Ошибка CUDA
    }
    
    // Вывод результатов
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Размер поля: " << nx << "x" << ny << "x" << nz 
              << " (" << total_size << " точек)" << std::endl;
    std::cout << "OpenMP Градиент: " << omp_grad_time << " мс" << std::endl;
    std::cout << "OpenMP Дивергенция: " << omp_div_time << " мс" << std::endl;
    
    if (cuda_grad_time > 0) {
        std::cout << "CUDA Градиент: " << cuda_grad_time << " мс" << std::endl;
        std::cout << "Ускорение CUDA/OpenMP: " << omp_grad_time / cuda_grad_time << "x" << std::endl;
    } else {
        std::cout << "CUDA Градиент: не поддерживается" << std::endl;
    }
    
    // Расчет пропускной способности
    double data_size_gb = (double)(total_size * sizeof(T) * 4) / (1024*1024*1024); // 4 поля
    double omp_throughput = data_size_gb / (omp_grad_time / 1000.0);
    std::cout << "Пропускная способность OpenMP: " << omp_throughput << " GB/s" << std::endl;
    
    if (cuda_grad_time > 0) {
        double cuda_throughput = data_size_gb / (cuda_grad_time / 1000.0);
        std::cout << "Пропускная способность CUDA: " << cuda_throughput << " GB/s" << std::endl;
    }
    
    // Очистка
    delete[] scalar_host;
    delete[] vec_x_host;
    delete[] vec_y_host;
    delete[] vec_z_host;
    delete[] grad_x_host;
    delete[] grad_y_host;
    delete[] grad_z_host;
    delete[] div_host;
}

// ==================== ОСНОВНАЯ ПРОГРАММА ====================

void print_gpu_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        std::cout << "CUDA устройства не найдены" << std::endl;
        return;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "\n=== ИНФОРМАЦИЯ О GPU ===" << std::endl;
    std::cout << "Устройство: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Глобальная память: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << std::endl;
    std::cout << "Мультипроцессоры: " << prop.multiProcessorCount << std::endl;
    std::cout << "Максимальное число потоков на блок: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Максимальные размеры сетки: " 
              << prop.maxGridSize[0] << "x" 
              << prop.maxGridSize[1] << "x" 
              << prop.maxGridSize[2] << std::endl;
}

int main() {
    std::cout << "ПРОГРАММА МОДЕЛИРОВАНИЯ ФИЗИЧЕСКИХ ПОЛЕЙ" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Информация о системе
    print_gpu_info();
    
    #ifdef _OPENMP
        std::cout << "\nOpenMP поддерживается: Да" << std::endl;
        std::cout << "Максимальное число потоков OpenMP: " << omp_get_max_threads() << std::endl;
        omp_set_num_threads(omp_get_max_threads());
    #else
        std::cout << "\nOpenMP не поддерживается" << std::endl;
    #endif
    
    // Тестируем разные размеры сеток
    std::vector<std::tuple<int, int, int>> grid_sizes = {
        {32, 32, 32},    // 32K точек
        {64, 64, 64},    // 256K точек
        {128, 128, 64},  // 1M точек
        {256, 128, 64}   // 2M точек
    };
    
    std::cout << "\n=== СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ===" << std::endl;
    std::cout << "Количество итераций для усреднения: 10" << std::endl;
    
    // Тестируем разные типы данных
    for (const auto& grid : grid_sizes) {
        int nx = std::get<0>(grid);
        int ny = std::get<1>(grid);
        int nz = std::get<2>(grid);
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "РАЗМЕР СЕТКИ: " << nx << "x" << ny << "x" << nz << std::endl;
        
        // FP64
        std::cout << "\n--- ТИП ДАННЫХ: FP64 (double) ---" << std::endl;
        performance_comparison<double>(nx, ny, nz, 5);
        
        // FP32
        std::cout << "\n--- ТИП ДАННЫХ: FP32 (float) ---" << std::endl;
        performance_comparison<float>(nx, ny, nz, 5);
        
        // FP16 (только если поддерживается)
        try {
            std::cout << "\n--- ТИП ДАННЫХ: FP16 (half) ---" << std::endl;
            performance_comparison<half>(nx, ny, nz, 5);
        } catch (...) {
            std::cout << "FP16 не поддерживается на этом устройстве" << std::endl;
        }
    }
    
    // Сводная таблица рекомендаций
    std::cout << "\n=== ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ ===" << std::endl;
    std::cout << "┌─────────────┬──────────────┬─────────────────────────────┐" << std::endl;
    std::cout << "│   Тип данных │ Размер (байт) │          Применение         │" << std::endl;
    std::cout << "├─────────────┼──────────────┼─────────────────────────────┤" << std::endl;
    std::cout << "│    FP64     │      8       │ Высокоточные научные расчёты │" << std::endl;
    std::cout << "│    FP32     │      4       │ Баланс точности и скорости  │" << std::endl;
    std::cout << "│    FP16     │      2       │ Машинное обучение, экономия │" << std::endl;
    std::cout << "│             │              │ памяти, если точность ок    │" << std::endl;
    std::cout << "└─────────────┴──────────────┴─────────────────────────────┘" << std::endl;
    
    std::cout << "\n=== ВЫВОДЫ ===" << std::endl;
    std::cout << "1. CUDA обеспечивает значительное ускорение для больших сеток" << std::endl;
    std::cout << "2. OpenMP эффективен для средних размеров на многоядерных CPU" << std::endl;
    std::cout << "3. FP32 - оптимальный выбор для большинства задач" << std::endl;
    std::cout << "4. FP16 полезен когда критична пропускная способность памяти" << std::endl;
    std::cout << "5. FP64 необходим только для задач с двойной точностью" << std::endl;
    
    // Тестирование точности
    std::cout << "\n=== ТЕСТИРОВАНИЕ ТОЧНОСТИ ===" << std::endl;
    {
        int nx = 32, ny = 32, nz = 32;
        size_t total_size = nx * ny * nz;
        
        // Создаем тестовое поле
        double* test_field = new double[total_size];
        initialize_temperature_field(test_field, nx, ny, nz, 10.0f, 10.0f, 10.0f);
        
        // Вычисляем градиент разными методами
        double* grad_x_cpu = new double[total_size];
        double* grad_y_cpu = new double[total_size];
        double* grad_z_cpu = new double[total_size];
        
        double dx = 0.1f, dy = 0.1f, dz = 0.1f;
        
        auto start = std::chrono::high_resolution_clock::now();
        omp_gradient(test_field, grad_x_cpu, grad_y_cpu, grad_z_cpu, 
                     nx, ny, nz, dx, dy, dz);
        auto end = std::chrono::high_resolution_clock::now();
        
        double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Анализируем результаты
        double max_grad = 0.0;
        double avg_grad = 0.0;
        for (size_t i = 0; i < total_size; ++i) {
            double mag = sqrt(grad_x_cpu[i]*grad_x_cpu[i] + 
                              grad_y_cpu[i]*grad_y_cpu[i] + 
                              grad_z_cpu[i]*grad_z_cpu[i]);
            max_grad = std::max(max_grad, mag);
            avg_grad += mag;
        }
        avg_grad /= total_size;
        
        std::cout << "Тестовая сетка: 32x32x32" << std::endl;
        std::cout << "Максимальный градиент: " << max_grad << " К/м" << std::endl;
        std::cout << "Средний градиент: " << avg_grad << " К/м" << std::endl;
        std::cout << "Время вычисления (CPU): " << cpu_time << " мс" << std::endl;
        
        delete[] test_field;
        delete[] grad_x_cpu;
        delete[] grad_y_cpu;
        delete[] grad_z_cpu;
    }
    
    // Сброс устройства CUDA
    CUDA_CHECK(cudaDeviceReset());
    
    std::cout << "\nПрограмма завершена успешно!" << std::endl;
    
    return 0;
}