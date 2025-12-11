@echo off
chcp 65001 > nul
title Менеджер CUDA программ
color 0F

:menu
cls
echo =========================================
echo         МЕНЕДЖЕР CUDA ПРОГРАММ
echo =========================================
echo.
echo [1] Полный тест производительности типов данных
echo [2] Определить информацию о GPU
echo [3] Быстрая компиляция (по умолчанию)
echo [4] Очистить временные файлы
echo [5] Проверить установку CUDA
echo [6] Выход
echo.
set /p choice="Выберите действие [1-6]: "

if "%choice%"=="1" goto option1
if "%choice%"=="2" goto option2
if "%choice%"=="3" goto option3
if "%choice%"=="4" goto option4
if "%choice%"=="5" goto option5
if "%choice%"=="6" goto exit
echo Неверный выбор. Попробуйте снова.
pause
goto menu

:option1
call run_cuda_test.bat
goto menu

:option2
call gpu_query.bat
goto menu

:option3
call compile_simple.bat
goto menu

:option4
call cleanup.bat
goto menu

:option5
cls
echo =========================================
echo     ПРОВЕРКА УСТАНОВКИ CUDA
echo =========================================
echo.
where nvcc >nul 2>nul
if %errorlevel% equ 0 (
    echo ✓ NVCC найден
    nvcc --version | findstr "release"
) else (
    echo ✗ NVCC не найден
)

echo.
where cuda-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo ✓ nvidia-smi найден
) else (
    echo ✗ nvidia-smi не найден
)

echo.
set CUDA_PATH
if errorlevel 1 (
    echo ✗ Переменная CUDA_PATH не установлена
) else (
    echo ✓ CUDA_PATH: %CUDA_PATH%
)

echo.
echo Для установки CUDA Toolkit:
echo 1. Скачайте с https://developer.nvidia.com/cuda-downloads
echo 2. Выберите версию для Windows
echo 3. Установите Visual Studio 2019/2022 перед установкой CUDA
echo.
pause
goto menu

:exit
echo.
echo До свидания!
pause
exit /b 0