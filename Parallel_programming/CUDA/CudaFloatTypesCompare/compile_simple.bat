@echo off
chcp 65001 > nul
title Простая компиляция CUDA программы
color 0E

echo ======================================
echo   ПРОСТАЯ КОМПИЛЯЦИЯ CUDA ПРОГРАММЫ
echo ======================================
echo.

REM Проверка NVCC
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ОШИБКА: NVCC не найден. Проверьте установку CUDA Toolkit.
    pause
    exit /b 1
)

REM Настройки по умолчанию
set DEFAULT_ARCH=sm_75
set FILENAME=main.cu
set OUTPUT=fp_test.exe

if not exist "%FILENAME%" (
    echo ОШИБКА: Файл %FILENAME% не найден!
    echo Создайте файл main.cu или используйте run_cuda_test.bat
    pause
    exit /b 1
)

echo Используется архитектура по умолчанию: %DEFAULT_ARCH%
echo Файл для компиляции: %FILENAME%
echo Выходной файл: %OUTPUT%
echo.

echo Начинаю компиляцию...
echo.

REM Компиляция
nvcc -std=c++17 -O3 -arch=%DEFAULT_ARCH% -o %OUTPUT% %FILENAME%

if %errorlevel% neq 0 (
    echo.
    echo ОШИБКА КОМПИЛЯЦИИ!
    echo Попробуйте изменить архитектуру:
    echo   sm_70  - для Tesla V100, GTX 10xx
    echo   sm_75  - для RTX 20xx
    echo   sm_80  - для RTX 30xx, A100
    echo   sm_86  - для RTX 30xx mobile
    echo.
    pause
    exit /b 1
)

echo.
echo Компиляция успешно завершена!
echo.
echo Запуск программы...
echo ======================================
echo.

%OUTPUT%

echo.
echo ======================================
echo Программа завершена.
echo Для повторного запуска выполните: %OUTPUT%
echo.

pause