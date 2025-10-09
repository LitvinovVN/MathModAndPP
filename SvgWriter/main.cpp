#include "SvgColors.hpp"
#include "SvgWriter.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace SVG;

void example1()
{
    // Создаем SVG документ размером 800x600

    SvgWriter svg(800, 600, SvgColorConverter::toString(SvgColor::LIGHT_BLUE));
    
    // Создаем различные стили
    Style redFillStyle;
    redFillStyle.setFill(Color(255, 0, 0))
                .setStroke(Color(0, 0, 0))
                .setStrokeWidth(2);
    
    Style blueStrokeStyle;
    blueStrokeStyle.setStroke(Color(0, 0, 255))
                   .setStrokeWidth(3)
                   .setFill(Color::fromHex("#00000000")); // Прозрачная заливка
    
    Style greenStyle;
    greenStyle.setFill(Color(0, 255, 0, 128))  // Полупрозрачный зеленый
              .setStroke(Color(0, 100, 0))
              .setStrokeWidth(1)
              .setFillOpacity(0.5);
    
    // Добавляем прямоугольник
    svg.addRectangle(50, 50, 200, 100, redFillStyle);
    
    // Добавляем круг
    svg.addCircle(400, 300, 80, blueStrokeStyle);
    
    // Добавляем линию
    svg.addLine(100, 400, 700, 400, blueStrokeStyle);
    
    // Добавляем ломаную линию
    std::vector<Point> polylinePoints = {
        Point(600, 100), Point(650, 150), Point(700, 120), 
        Point(750, 180), Point(780, 140)
    };
    svg.addPolyline(polylinePoints, blueStrokeStyle);
    
    // Добавляем многоугольник (треугольник)
    std::vector<Point> polygonPoints = {
        Point(200, 300), Point(250, 400), Point(150, 400)
    };
    svg.addPolygon(polygonPoints, greenStyle);
    
    // Добавляем текст
    Style textStyle;
    textStyle.setFill(Color(0, 0, 0))
             .setStroke(Color(255, 255, 255))
             .setStrokeWidth(0.5);
    
    svg.addText(300, 500, "Hello SVG World!", textStyle, "Verdana", 24);
    
    // Создаем простой график
    Style graphStyle;
    graphStyle.setStroke(Color(255, 0, 0))
              .setStrokeWidth(2);
    
    std::vector<Point> sineWave;
    for (int x = 0; x < 800; x += 10) {
        double y = 200 + 50 * std::sin(x * 0.02);
        sineWave.emplace_back(x, y);
    }
    svg.addPolyline(sineWave, graphStyle);
    
    // Сохраняем в файл
    if (svg.saveToFile("example1.svg")) {
        std::cout << "SVG file created: example1.svg" << std::endl;
    } else {
        std::cerr << "SVG file creating error" << std::endl;
        //return 1;
    }
}

void example2_vectors()
{
    // Создаем SVG документ
    SvgWriter svg(800, 600, "white");
    
    // Стили для векторов
    Style redVectorStyle;
    redVectorStyle.setStroke(Color(255, 0, 0))
                  .setStrokeWidth(2)
                  .setFill(Color(255, 0, 0));
    
    Style blueVectorStyle;
    blueVectorStyle.setStroke(Color(0, 0, 255))
                   .setStrokeWidth(3)
                   .setFill(Color(0, 0, 255));
    
    Style greenVectorStyle;
    greenVectorStyle.setStroke(Color(0, 255, 0))
                    .setStrokeWidth(1.5)
                    .setFill(Color(0, 255, 0, 128)); // Полупрозрачная заливка
    
    // Разные стили для линии и стрелки
    Style lineStyle;
    lineStyle.setStroke(Color(0, 0, 0))
             .setStrokeWidth(2);
    
    Style arrowStyle;
    arrowStyle.setFill(Color(255, 0, 0))
              .setStroke(Color(0, 0, 0))
              .setStrokeWidth(1);
    
    // Добавляем векторы с разными параметрами
    
    // Простой красный вектор
    svg.addVector(100, 100, 200, 150, redVectorStyle, 10);
    
    // Синий вектор с увеличенной стрелкой
    svg.addVector(100, 200, 250, 200, blueVectorStyle, 15);
    
    // Зеленый вектор с измененным углом стрелки
    svg.addVector(100, 300, 200, 250, greenVectorStyle, 12, 45);
    
    // Вектор с разными стилями для линии и стрелки
    svg.addVector(100, 400, 250, 300, lineStyle, arrowStyle);
    
    // Создаем векторное поле (градиент)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            double x = 350 + i * 30;
            double y = 100 + j * 30;
            
            // Векторы, указывающие от центра
            double centerX = 550;
            double centerY = 250;
            double dx = x - centerX;
            double dy = y - centerY;
            double length = std::sqrt(dx * dx + dy * dy);
            
            if (length > 0) {
                // Нормализуем и масштабируем
                double scale = 20.0 / length;
                double endX = x + dx * scale;
                double endY = y + dy * scale;
                
                Style fieldStyle;
                fieldStyle.setStroke(Color(128, 0, 128))
                          .setStrokeWidth(1)
                          .setFill(Color(128, 0, 128));
                
                svg.addVector(x, y, endX, endY, fieldStyle, 5);
            }
        }
    }
    
    // Добавляем координатные оси с векторами
    Style axisStyle;
    axisStyle.setStroke(Color(0, 0, 0))
             .setStrokeWidth(1)
             .setFill(Color(0, 0, 0));
    
    // Ось X
    svg.addVector(50, 500, 750, 500, axisStyle, 8); // Стрелка оси X
    
    // Ось Y
    svg.addVector(50, 500, 50, 50, axisStyle, 8); // Стрелка оси Y
    
    // Подписи осей
    Style textStyle;
    textStyle.setFill(Color(0, 0, 0))
             .setStroke(Color(255, 255, 255))
             .setStrokeWidth(0.1);
    
    svg.addText(760, 505, "X", textStyle, "Times New Roman", 16);
    svg.addText(45, 40, "Y", textStyle, "Times New Roman", 16);
    
    // Сохраняем в файл
    if (svg.saveToFile("example2_vectors.svg")) {
        std::cout << "SVG file created: example2_vectors.svg" << std::endl;
    } else {
        std::cerr << "SVG file creating error" << std::endl;
        //return 1;
    }
}

int main() {
    example1();
    example2_vectors();
    
    return 0;
}