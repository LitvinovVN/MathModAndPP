// g++ heat2d.cpp -o heat2d
// ./heat2d
#include <iostream>
#include <fstream>
using namespace std;

struct TGridConfig
{
    size_t Nx;
    size_t Ny;
};

TGridConfig TGridConfig_ReadFromFile(string fileName)
{
    TGridConfig conf;
    conf.Nx = 100;
    conf.Ny = 150;

    return conf;
}

int main()
{
    cout << "2D Heat Equation" << endl;
    ofstream fout("report.txt");

    // 1. Считываем описание геометрии из текстового файла geometry2d.txt

    // 2. Считываем параметры сетки из текстового файла config.txt
    TGridConfig gridConf = TGridConfig_ReadFromFile("geometry2d.txt");
    fout << "Nx = " << gridConf.Nx << endl;
    fout << "Ny = " << gridConf.Ny << endl;
    

    // 3. Формируем равномерную расчетную сетку

    // 4. Расчет по явной схеме последовательно на CPU

    // 5. Расчет по явной схеме параллельно на CPU

    // 6. Расчет по явной схеме последовательно на GPU

    // 7. Расчет по явной схеме параллельно на GPU

    // 8. Метод расщепления по пространственным направлениям

    // 9. Схема с весами

    // 10. Метод Зейделя
}