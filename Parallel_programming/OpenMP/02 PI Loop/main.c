/*
This program will numerically compute the integral of
                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.
The program was parallelized using OpenMP by adding just
four lines
(1) A line to include omp.h -- the include file that
contains OpenMP's function prototypes and constants.
(2) A pragma that tells OpenMP to create a team of threads
(3) A pragma to cause one of the threads to print the
number of threads being used by the program.
(4) A pragma to split up loop iterations among the team
of threads.  This pragma includes 2 clauses to (1) create a
private variable and (2) to cause the threads to compute their
sums locally and then combine their local sums into a
single global value.
History: Written by Tim Mattson, 11/99.

02. Разработать программу, численно вычисляющую интеграл выражения
                   4/(1+х*х)
от 0 до 1. Значение этого интеграла равно пи, что позволяет выполнить проверку результата вычислений.
Программу необходимо распараллелить с использованием технологии OpenMP путем добавления только
четыре линии
(1) #include <omp.h> -- подключаем файл omp.h,
содержащий прототипы функций и константы библиотеки OpenMP.
(2) Прагма, которая указывает OpenMP создать группу потоков.
(3) Прагма, заставляющая один из потоков печатать
количество потоков, используемых программой.
(4) Прагма для разделения итераций цикла между командой
потоков. Эта прагма создает частная переменная и вычисляет свои
суммы локально, а затем объединяет их локальные суммы в
единое глобальное значение.

gcc -fopenmp main.c -o app
./app
*/

#include <stdio.h> // (1)
#include <omp.h>
static long num_steps = 100000000;
double step;
int main()
{
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;

    step = 1.0 / (double)num_steps;
    for (i = 1; i <= 4; i++)
    {
        sum = 0.0;
        omp_set_num_threads(i);
        start_time = omp_get_wtime();
        #pragma omp parallel // (2)
        {
            #pragma omp single // (3)
            printf(" num_threads = %d", omp_get_num_threads());

            #pragma omp for reduction(+: sum)    // (4)
            for (i = 1; i <= num_steps; i++)
            {
                x = (i - 0.5) * step;
                sum = sum + 4.0 / (1.0 + x * x);
            }
        }
        pi = step * sum;
        run_time = omp_get_wtime() - start_time;
        printf("\n pi is %f in %f seconds and %d threads\n", pi, run_time, i);
    }
}