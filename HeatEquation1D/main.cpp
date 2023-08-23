#include <iostream>
#include <fstream>

void config()
{
    setlocale(LC_ALL, "");
    std::wcout<<L"Уравнение теплопроводности 1D"<< std::endl;
}

int main()
{
    config();

    // Количество пространственных узлов
    int N = 101;
    // Время моделирования, сек.
    double t_end = 60;
    // Толщина пластины, м.
    double L = 0.1;
    // Коэффициент теплопроводности материала пластины, Вт/(м гр.Цельс.)
    double lambda = 46;
    // Плотность материала пластины, кг/м.куб.
    double ro = 7800;
    // Теплоёмкость материала пластины, Дж/(кг. град.Цельс.)
    double c = 460;

    double T0 = 20;
    double Tl = 300;
    double Tr = 100;

    double h = L/(N-1);
    std::wcout<<L"h = "<< h << L" м." << std::endl;

    double tau = t_end/100;
    std::wcout<<L"tau = "<< tau <<L" сек." << std::endl;

    double* T = new double[N];
    double* alfa = new double[N];
    double* beta = new double[N];

    std::wcout<<L"Выделено памяти: "<< 3 * N * sizeof(double) << L" байт." << std::endl;

    // Инициализируем поле температуры в начальный момент времени
    for (size_t i = 0; i < N; i++)
    {
        T[i] = T0;
    }
        
    T[0] = Tl;
    T[N-1] = Tr;

    /*for (size_t i = 0; i < N; i++)
    {
        std::wcout << T[i] << " ";
    }*/

    double time = 0;
    while (time < t_end)
    {
        time += tau;

        alfa[0] = 0;
        beta[0] = Tl;

        for (size_t i = 1; i < N-1; i++)
        {
            double ai = lambda / (h*h);
            double bi = 2 * lambda / (h*h) + ro * c / tau;
            double ci = lambda / (h*h);
            double fi = - ro * c * T[i] / tau;
            alfa[i] = ai / (bi - ci * alfa[i-1]);
            beta[i] = (ci * beta[i-1] - fi) / (bi - ci * alfa[i-1]);
        }
        
        for (size_t i = N-2; i > 0; i--)
        {
            T[i] = alfa[i] * T[i+1] + beta[i];
        }                
    }
    
    
    /*for (size_t i = 0; i < N; i++)
    {
        std::wcout << i*h << " " << T[i] << std::endl;
    }*/

    std::ofstream out("out.txt");          // поток для записи в файл
    for (size_t i = 0; i < N; i++)
    {
        out << i*h << " " << T[i] << std::endl;
    }

    delete[] T;
    delete[] alfa;
    delete[] beta;

    return 0;
}