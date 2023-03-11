// Программа для генерирования файлов с данными
// (для трёхмерной расчетной сетки)

// Запуск:
// nvcc grid3d.cpp -o ../bin/grid3d
// ../bin/grid3d
#include <iostream>
#include <fstream>
#include <string>

enum DataType {R = 1, U, V, W, C0, C1, C2, C3, C4, C5, C6};

// Структура "Файлы данных"
struct DataFiles
{
    // Тип данных
    DataType dataType;

    // Кол-во узлов расчетной сетки
    // по пространственным координатам x, y, z
    int nx = -1;
    int ny = -1;
    int nz = -1;
    // Каталог
    std::string folderName = "";

    double* read_X_line(int indZ, int indY, DataType dt = R)
    {
        double* arr = new double[nx];
        
        // Считываем файл с параметрами сетки
        std::string fileName = folderName + std::to_string(indZ) + ".dat";;
        std::ifstream fs_conf(fileName);
        
        for(int j = 0;  j <= indY; j++)
            for(int i = 0; i < nx; i++)
            {
                double val;
                fs_conf >> val;
                if(j==indY)
                    arr[i] = val;
            }       

        return arr;
    }

    void print()
    {
        std::cout << nx << " " << ny << " " << nz << " " << dataType << std::endl;
    }

    void open(std::string folderName)
    {
        this->folderName = folderName;
        // Считываем файл с параметрами сетки
        std::string fileName = folderName + "grid.conf";
        std::ifstream fs_conf(fileName);
        fs_conf >> nx;
        fs_conf >> ny;
        fs_conf >> nz;
        int dt;
        fs_conf >> dt;
        if(dt==1)
            dataType = R;

        fs_conf.close();
        std::cout << folderName << " conf readed" << std::endl;
    }

    void writeData()
    {
        // Записываем файл с параметрами сетки
        std::string fileName = folderName + "grid.conf";
        std::ofstream fs_conf(fileName);
        fs_conf << nx << " " << ny << " " << nz << " " << dataType;
        fs_conf.close();
        std::cout << folderName << " conf created" << std::endl;

        // Записываем файлы с данными
        for (int k = 0; k < nz; k++)
        {
            std::string fileName = folderName + std::to_string(k) + ".dat";
            //std::cout << fileName << std::endl;
            std::ofstream fs_r(fileName);
            //std::cout << fs_r.is_open() << std::endl;
            for (int j = 0; j < ny; j++)
            {
                for (int i = 0; i < nx; i++)
                {
                    fs_r << i + 0.1 * j + 0.01 * k;
                    if(i < nx - 1)
                        fs_r << " ";
                }
                if(j < ny - 1)
                    fs_r << std::endl;
            }
        }
        std::cout << folderName << " created" << std::endl;
    }
};


int main()
{
    std::cout << "--- grid3d ---" << std::endl;       

    DataFiles dfc;
    dfc.nx = 30;
    dfc.ny = 20;
    dfc.nz = 10;
    dfc.dataType = R;
    dfc.folderName = "../data/r/";

    dfc.writeData(); 
    
    DataFiles dfc2;
    dfc2.open("../data/r/");
    dfc2.print();
    auto arr_k1_y2 = dfc2.read_X_line(1,2);

    for(int i = 0; i < dfc2.nx; i++)
    {
        std::cout << arr_k1_y2[i] << " ";
    }

    return 0;
}