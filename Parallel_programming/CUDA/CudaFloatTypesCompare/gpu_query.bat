@echo off
chcp 65001 > nul
title CUDA GPU Information
color 0B

echo ========================================
echo      ИНФОРМАЦИЯ О GPU И КОМПИЛЯЦИЯ
echo ========================================
echo.

where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ОШИБКА: Компилятор NVCC не найден!
    pause
    exit /b 1
)

echo Создание программы для определения GPU...
(
echo #include ^<cuda_runtime.h^>
echo #include ^<iostream^>
echo int main^(^) {
echo     int deviceCount = 0;
echo     cudaGetDeviceCount^(^&deviceCount^);
echo     if ^(deviceCount == 0^) {
echo         std::cout ^<^< "CUDA устройства не найдены!" ^<^< std::endl;
echo         return 0;
echo     }
echo     for ^(int device = 0; device ^< deviceCount; device++^) {
echo         cudaDeviceProp prop;
echo         cudaGetDeviceProperties^(^&prop, device^);
echo         std::cout ^<^< "Устройство " ^<^< device ^<^< ": " ^<^< prop.name ^<^< std::endl;
echo         std::cout ^<^< "  Compute Capability: " ^<^< prop.major ^<^< "." ^<^< prop.minor ^<^< std::endl;
echo         std::cout ^<^< "  Адаптер архитектуры: ";
echo         // Определение архитектуры
echo         if ^(prop.major == 9^) {
echo             if ^(prop.minor == 0^) std::cout ^<^< "Hopper";
echo         } else if ^(prop.major == 8^) {
echo             if ^(prop.minor == 9^) std::cout ^<^< "Ada Lovelace";
echo             else if ^(prop.minor == 7^) std::cout ^<^< "Ampere ^(Orin^)";
echo             else if ^(prop.minor == 6^) std::cout ^<^< "Ampere ^(мобильные^)";
echo             else if ^(prop.minor == 0^) std::cout ^<^< "Ampere";
echo         } else if ^(prop.major == 7^) {
echo             if ^(prop.minor == 5^) std::cout ^<^< "Turing";
echo             else if ^(prop.minor == 0^) std::cout ^<^< "Volta";
echo         } else if ^(prop.major == 6^) {
echo             if ^(prop.minor == 1^) std::cout ^<^< "Pascal ^(GP102^)";
echo             else if ^(prop.minor == 0^) std::cout ^<^< "Pascal ^(GP100^)";
echo         } else if ^(prop.major == 5^) {
echo             std::cout ^<^< "Maxwell";
echo         }
echo         std::cout ^<^< std::endl;
echo         std::cout ^<^< "  Глобальная память: " ^<^< prop.totalGlobalMem / 1024 / 1024 / 1024 ^<^< " GB" ^<^< std::endl;
echo         std::cout ^<^< "  Мультипроцессоры: " ^<^< prop.multiProcessorCount ^<^< std::endl;
echo         std::cout ^<^< "  Архитектура для компиляции: sm_" ^<^< prop.major ^<^< prop.minor ^<^< std::endl;
echo         std::cout ^<^< std::endl;
echo     }
echo     return 0;
echo }
) > gpu_query.cu

echo Компиляция программы определения GPU...
nvcc -o gpu_query.exe gpu_query.cu

if exist gpu_query.exe (
    echo.
    echo =========== РЕЗУЛЬТАТЫ ===========
    echo.
    gpu_query.exe
    
    echo.
    echo РЕКОМЕНДАЦИИ ДЛЯ КОМПИЛЯЦИИ:
    echo.
    
    REM Анализ результатов
    gpu_query.exe > gpu_info.txt 2>&1
    
    for /f "tokens=*" %%a in ('findstr /C:"Архитектура для компиляции" gpu_info.txt') do (
        echo Для лучшей производительности используйте:
        echo   nvcc -arch=%%a
        echo.
    )
    
    del gpu_info.txt
    del gpu_query.exe
    del gpu_query.cu
) else (
    echo Не удалось скомпилировать программу определения GPU
)

echo.
pause