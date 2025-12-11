#!/bin/bash

# Создаем директорию для сборки
mkdir -p build
cd build

# Конфигурируем CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Компилируем
make -j$(nproc)

# Проверяем, что файл создан
if [ -f "fp_comparison" ]; then
    echo "Запуск программы..."
    echo ""
    ./fp_comparison
else
    echo "Ошибка: исполняемый файл не найден!"
    exit 1
fi