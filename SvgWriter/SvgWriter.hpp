#ifndef SVGWRITER_H
#define SVGWRITER_H

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <cmath>

namespace SVG {

// Базовый класс для всех SVG элементов
class SvgElement {
public:
    virtual ~SvgElement() = default;
    virtual std::string toString() const = 0;
};

// Класс для представления цвета
class Color {
public:
    Color() : r(0), g(0), b(0), a(255) {}
    Color(int red, int green, int blue, int alpha = 255) 
        : r(red), g(green), b(blue), a(alpha) {}
    
    static Color fromHex(const std::string& hex) {
        if (hex.empty() || hex[0] != '#') return Color();
        
        int red, green, blue;
        if (hex.length() == 7) {
            sscanf(hex.c_str() + 1, "%02x%02x%02x", &red, &green, &blue);
            return Color(red, green, blue);
        }
        return Color();
    }
    
    std::string toHex() const {
        char buffer[8];
        snprintf(buffer, sizeof(buffer), "#%02x%02x%02x", r, g, b);
        return std::string(buffer);
    }
    
    std::string toString() const {
        if (a == 255) return toHex();
        return "rgba(" + std::to_string(r) + "," + std::to_string(g) + 
               "," + std::to_string(b) + "," + std::to_string(a/255.0) + ")";
    }

private:
    int r, g, b, a;
};

// Класс для точки/координаты
class Point {
public:
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    
    double getX() const { return x; }
    double getY() const { return y; }
    void setX(double newX) { x = newX; }
    void setY(double newY) { y = newY; }
    
    std::string toString() const {
        return std::to_string(x) + "," + std::to_string(y);
    }

private:
    double x, y;
};

// Стиль отрисовки
class Style {
public:
    Style() : strokeWidth(1), fillOpacity(1.0), strokeOpacity(1.0) {}
    
    // Setters with fluent interface
    Style& setFill(const Color& color) { fill = color; return *this; }
    Style& setStroke(const Color& color) { stroke = color; return *this; }
    Style& setStrokeWidth(double width) { strokeWidth = width; return *this; }
    Style& setFillOpacity(double opacity) { fillOpacity = opacity; return *this; }
    Style& setStrokeOpacity(double opacity) { strokeOpacity = opacity; return *this; }
    
    std::string toString() const {
        std::stringstream ss;
        
        if (fill.toString() != "rgba(0,0,0,1.000000)") {
            ss << "fill:" << fill.toString() << ";";
        } else {
            ss << "fill:none;";
        }
        
        if (stroke.toString() != "rgba(0,0,0,1.000000)") {
            ss << "stroke:" << stroke.toString() << ";";
            ss << "stroke-width:" << std::to_string(strokeWidth) << ";";
        }
        
        if (fillOpacity != 1.0) {
            ss << "fill-opacity:" << std::to_string(fillOpacity) << ";";
        }
        
        if (strokeOpacity != 1.0) {
            ss << "stroke-opacity:" << std::to_string(strokeOpacity) << ";";
        }
        
        return ss.str();
    }

private:
    Color fill;
    Color stroke;
    double strokeWidth;
    double fillOpacity;
    double strokeOpacity;
};

// Конкретные SVG элементы
class Rectangle : public SvgElement {
public:
    Rectangle(double x, double y, double width, double height, const Style& style = Style())
        : x(x), y(y), width(width), height(height), style(style) {}
    
    std::string toString() const override {
        std::stringstream ss;
        ss << "<rect x=\"" << x << "\" y=\"" << y 
           << "\" width=\"" << width << "\" height=\"" << height 
           << "\" style=\"" << style.toString() << "\" />";
        return ss.str();
    }

private:
    double x, y, width, height;
    Style style;
};

class Circle : public SvgElement {
public:
    Circle(double cx, double cy, double r, const Style& style = Style())
        : cx(cx), cy(cy), r(r), style(style) {}
    
    std::string toString() const override {
        std::stringstream ss;
        ss << "<circle cx=\"" << cx << "\" cy=\"" << cy 
           << "\" r=\"" << r << "\" style=\"" << style.toString() << "\" />";
        return ss.str();
    }

private:
    double cx, cy, r;
    Style style;
};

class Line : public SvgElement {
public:
    Line(double x1, double y1, double x2, double y2, const Style& style = Style())
        : x1(x1), y1(y1), x2(x2), y2(y2), style(style) {}
    
    std::string toString() const override {
        std::stringstream ss;
        ss << "<line x1=\"" << x1 << "\" y1=\"" << y1 
           << "\" x2=\"" << x2 << "\" y2=\"" << y2 
           << "\" style=\"" << style.toString() << "\" />";
        return ss.str();
    }

private:
    double x1, y1, x2, y2;
    Style style;
};

class Polyline : public SvgElement {
public:
    Polyline(const std::vector<Point>& points, const Style& style = Style(), bool close = false)
        : points(points), style(style), close(close) {}
    
    void addPoint(const Point& point) { points.push_back(point); }
    
    std::string toString() const override {
        if (points.empty()) return "";
        
        std::stringstream ss;
        ss << "<" << (close ? "polygon" : "polyline") << " points=\"";
        
        for (size_t i = 0; i < points.size(); ++i) {
            ss << points[i].toString();
            if (i < points.size() - 1) ss << " ";
        }
        
        ss << "\" style=\"" << style.toString() << "\" />";
        return ss.str();
    }

private:
    std::vector<Point> points;
    Style style;
    bool close;
};

class Text : public SvgElement {
public:
    Text(double x, double y, const std::string& content, const Style& style = Style(), 
         const std::string& fontFamily = "Arial", int fontSize = 12)
        : x(x), y(y), content(content), style(style), fontFamily(fontFamily), fontSize(fontSize) {}
    
    std::string toString() const override {
        std::stringstream ss;
        ss << "<text x=\"" << x << "\" y=\"" << y 
           << "\" font-family=\"" << fontFamily << "\" font-size=\"" << fontSize 
           << "\" style=\"" << style.toString() << "\">" << content << "</text>";
        return ss.str();
    }

private:
    double x, y;
    std::string content;
    Style style;
    std::string fontFamily;
    int fontSize;
};

// Элемент для отрисовки вектора со стрелкой
class VectorArrow : public SvgElement {
public:
    VectorArrow(double startX, double startY, double endX, double endY, 
                const Style& lineStyle = Style(), const Style& arrowStyle = Style(),
                double arrowheadSize = 10, double arrowheadAngle = 30)
        : start(startX, startY), end(endX, endY), lineStyle(lineStyle), 
          arrowStyle(arrowStyle), arrowheadSize(arrowheadSize), 
          arrowheadAngle(arrowheadAngle) {}
    
    VectorArrow(const Point& startPoint, const Point& endPoint,
                const Style& lineStyle = Style(), const Style& arrowStyle = Style(),
                double arrowheadSize = 10, double arrowheadAngle = 30)
        : start(startPoint), end(endPoint), lineStyle(lineStyle), 
          arrowStyle(arrowStyle), arrowheadSize(arrowheadSize), 
          arrowheadAngle(arrowheadAngle) {}
    
    std::string toString() const override {
        std::stringstream ss;
        
        // Рисуем линию вектора
        ss << "<line x1=\"" << start.getX() << "\" y1=\"" << start.getY() 
           << "\" x2=\"" << end.getX() << "\" y2=\"" << end.getY() 
           << "\" style=\"" << lineStyle.toString() << "\" />";
        
        // Вычисляем точки для стрелки
        auto arrowPoints = calculateArrowhead();
        
        // Рисуем стрелку как полигон
        ss << "<polygon points=\"";
        for (size_t i = 0; i < arrowPoints.size(); ++i) {
            ss << arrowPoints[i].toString();
            if (i < arrowPoints.size() - 1) ss << " ";
        }
        ss << "\" style=\"" << arrowStyle.toString() << "\" />";
        
        return ss.str();
    }

private:
    std::vector<Point> calculateArrowhead() const {
        std::vector<Point> points;
        
        // Вектор направления
        double dx = end.getX() - start.getX();
        double dy = end.getY() - start.getY();
        
        // Угол вектора
        double angle = std::atan2(dy, dx);
        
        // Углы для сторон стрелки
        double angle1 = angle + M_PI - arrowheadAngle * M_PI / 180.0;
        double angle2 = angle + M_PI + arrowheadAngle * M_PI / 180.0;
        
        // Вычисляем точки стрелки
        points.push_back(end);
        points.emplace_back(
            end.getX() + arrowheadSize * std::cos(angle1),
            end.getY() + arrowheadSize * std::sin(angle1)
        );
        points.emplace_back(
            end.getX() + arrowheadSize * std::cos(angle2),
            end.getY() + arrowheadSize * std::sin(angle2)
        );
        
        return points;
    }

private:
    Point start, end;
    Style lineStyle, arrowStyle;
    double arrowheadSize;
    double arrowheadAngle;
};

//////////////////////////////////////////////////////////

// Главный класс для создания SVG документов
class SvgWriter {
public:
    SvgWriter(int width, int height, const std::string& backgroundColor = "white")
        : width(width), height(height), backgroundColor(backgroundColor) {}
    
    ~SvgWriter() {
        if (file.is_open()) {
            file.close();
        }
    }
    
    // Методы для добавления элементов
    void addRectangle(double x, double y, double width, double height, 
                     const Style& style = Style()) {
        elements.push_back(std::make_unique<Rectangle>(x, y, width, height, style));
    }
    
    void addCircle(double cx, double cy, double r, const Style& style = Style()) {
        elements.push_back(std::make_unique<Circle>(cx, cy, r, style));
    }
    
    void addLine(double x1, double y1, double x2, double y2, const Style& style = Style()) {
        elements.push_back(std::make_unique<Line>(x1, y1, x2, y2, style));
    }
    
    void addPolyline(const std::vector<Point>& points, const Style& style = Style()) {
        elements.push_back(std::make_unique<Polyline>(points, style, false));
    }
    
    void addPolygon(const std::vector<Point>& points, const Style& style = Style()) {
        elements.push_back(std::make_unique<Polyline>(points, style, true));
    }
    
    void addText(double x, double y, const std::string& content, 
                const Style& style = Style(), const std::string& fontFamily = "Arial", 
                int fontSize = 12) {
        elements.push_back(std::make_unique<Text>(x, y, content, style, fontFamily, fontSize));
    }

    // Метод для добавления вектора со стрелкой
    void addVector(double startX, double startY, double endX, double endY,
                  const Style& lineStyle = Style(), const Style& arrowStyle = Style(),
                  double arrowheadSize = 10, double arrowheadAngle = 30) {
        elements.push_back(std::make_unique<VectorArrow>(
            startX, startY, endX, endY, lineStyle, arrowStyle, 
            arrowheadSize, arrowheadAngle
        ));
    }
    
    void addVector(const Point& start, const Point& end,
                  const Style& lineStyle = Style(), const Style& arrowStyle = Style(),
                  double arrowheadSize = 10, double arrowheadAngle = 30) {
        elements.push_back(std::make_unique<VectorArrow>(
            start, end, lineStyle, arrowStyle, arrowheadSize, arrowheadAngle
        ));
    }
    
    // Перегруженный метод с одним стилем для всей векторной стрелки
    void addVector(double startX, double startY, double endX, double endY,
                  const Style& style = Style(),
                  double arrowheadSize = 10, double arrowheadAngle = 30) {
        addVector(startX, startY, endX, endY, style, style, arrowheadSize, arrowheadAngle);
    }
    
    void addVector(const Point& start, const Point& end,
                  const Style& style = Style(),
                  double arrowheadSize = 10, double arrowheadAngle = 30) {
        addVector(start, end, style, style, arrowheadSize, arrowheadAngle);
    }
    
    // Сохранение в файл
    bool saveToFile(const std::string& filename) {
        file.open(filename);
        if (!file.is_open()) {
            return false;
        }
        
        writeHeader();
        writeElements();
        writeFooter();
        
        file.close();
        return true;
    }
    
    // Очистка документа
    void clear() {
        elements.clear();
    }
    
    // Геттеры
    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    void writeHeader() {
        file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        file << "<svg width=\"" << width << "\" height=\"" << height 
             << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
        
        if (backgroundColor != "none") {
            file << "<rect width=\"100%\" height=\"100%\" fill=\"" << backgroundColor << "\"/>\n";
        }
    }
    
    void writeElements() {
        for (const auto& element : elements) {
            file << element->toString() << "\n";
        }
    }
    
    void writeFooter() {
        file << "</svg>\n";
    }

private:
    int width, height;
    std::string backgroundColor;
    std::vector<std::unique_ptr<SvgElement>> elements;
    std::ofstream file;
};

} // namespace SVG

#endif // SVGWRITER_H