cmake_minimum_required(VERSION 3.10)
project(FloatingPointComparison)

# Настройка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Поиск CUDA
find_package(CUDA REQUIRED)

# Проверка версии CUDA
if(CUDA_VERSION VERSION_LESS 11.0)
    message(WARNING "Для полной функциональности рекомендуется CUDA 11.0+")
endif()

# Настройка флагов компиляции
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W3")
endif()

# Добавление исполняемого файла
cuda_add_executable(fp_comparison main.cu)

# Установка свойств CUDA
set_target_properties(fp_comparison PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89"  # Поддержка различных архитектур
)

# Целевая архитектура (можно изменить под свое оборудование)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")  # Volta
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")  # Turing
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")  # Ampere
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")  # Ampere (мобильные)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_89,code=sm_89")  # Ada Lovelace
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_90,code=sm_90")  # Hopper (для FP8)

message(STATUS "CUDA версия: ${CUDA_VERSION}")
message(STATUS "Компилятор CUDA: ${CUDA_NVCC_EXECUTABLE}")