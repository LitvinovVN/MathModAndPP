/* Задача 03. Создать структуру "Точка в трёхмерном пространстве".
 Инициализировать поля структуры.
 Вывести структуру в консоль

 Запуск:
 gcc main.c -o app
 ./app

*/
#include <conio.h>
#include <stdio.h>
#include <math.h>

//////////////////////////////////////////////////
struct point_t {
    int x;
    int y;
    int z;
};

//Определяем новый тип
typedef struct point_t Point3D;
 
//////////////////////////////////////////////////

void printPoint3D(Point3D point)
{
    printf("Point3D: { %d, %d, %d }\n", point.x, point.y, point.z);
}

Point3D createPoint3D(int x, int y, int z)
{
    Point3D point = {x, y, z};
    return point;
}

float calculateDistance(Point3D point)
{
    float distance = sqrt((float) (point.x*point.x + point.y*point.y + point.z*point.z));
    return distance;
}

//////////////////////////////////////////////////

void main() {    
    Point3D pointA = createPoint3D(5, 10, 15);
    printPoint3D(pointA);
    printf("distance = %.3f\n", calculateDistance(pointA));

    getch();
}