#pragma once

#include <iostream>

#include "../Geometry/_IncludeGeometry.hpp"

/// @brief Консольный пользовательский интерфейс для решения модельных задач
struct ModelProblems_ConsoleUI
{
    /// @brief Уравнение Пуассона в прямоугольнике с граничными условиями 1го рода
    static void Poisson2D_Rectangle_bc1111_ConsoleUI()
    {
        std::cout << "Poisson2D_Rectangle_bc1111_ConsoleUI()\n";

        // 1. Создаём объект, описывающий геометрию расчетной области
        // Прямоугольник 2*1
        IGeometry* rectangle = new G2DRectangle(2, 1);
        // Расположение в точке (1, 2)        
        IGeometryComposition* geomComp = new GeometryComposition2D();
        geomComp->Add(rectangle, 1, 2);
        geomComp->Print();
        // 2. Описываем граничные условия
        // 3. Задаём уравнение в непрерывной форме
        // Задаём искомую физическую величину
        // Задаём функцию правой части

        // Задаём параметры расчетной сетки
        // IGridParams* gridParams_ptr = new CalculationGrid2DUniformParams(0.1, 0.2);
        // IGrid* grid = GridFactory::Create2DUniformGrid(geomComp, gridParams_ptr)

        // Задаём схему дискретизации по пространству
        // Задаём схему дискретизации граничных условий

        
        // 8. Формируем СЛАУ
        // 9. Решаем СЛАУ
        // 10. Сохраняем результаты расчета в файл
        // 11. Создаём визуализацию
    }
};