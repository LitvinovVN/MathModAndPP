@echo off
chcp 65001 > nul
title Очистка временных файлов CUDA
color 0C

echo УДАЛЕНИЕ ВРЕМЕННЫХ ФАЙЛОВ CUDA
echo ================================
echo.

set /p confirm="Вы уверены, что хотите удалить все временные файлы? (Y/N): "

if /i "%confirm%" neq "Y" (
    echo Отменено.
    pause
    exit /b 0
)

echo.
echo Удаление файлов...

REM Удаление исполняемых файлов
if exist *.exe (
    echo Удаление *.exe...
    del /q *.exe
)

REM Удаление объектных файлов
if exist *.obj (
    echo Удаление *.obj...
    del /q *.obj
)

REM Удаление файлов библиотек
if exist *.lib (
    echo Удаление *.lib...
    del /q *.lib
)

if exist *.exp (
    echo Удаление *.exp...
    del /q *.exp
)

REM Удаление отладочных файлов
if exist *.pdb (
    echo Удаление *.pdb...
    del /q *.pdb
)

if exist *.ilk (
    echo Удаление *.ilk...
    del /q *.ilk
)

REM Удаление временных CUDA файлов
if exist *.cudafe1.* (
    echo Удаление временных CUDA файлов...
    del /q *.cudafe1.*
)

if exist *.fatbin.* (
    echo Удаление *.fatbin.*...
    del /q *.fatbin.*
)

if exist *.cubin (
    echo Удаление *.cubin...
    del /q *.cubin
)

REM Удаление файлов журналов
if exist nvcc*.log (
    echo Удаление логов nvcc...
    del /q nvcc*.log
)

echo.
echo Очистка завершена!
echo.

pause