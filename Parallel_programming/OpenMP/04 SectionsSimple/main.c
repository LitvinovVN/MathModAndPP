/*
   4. Gример распараллеливания программы, содержащей
последовательный вызов функций run_function1 и run_function2, которые
не зависят друг от друга (т.е. не используют общих данных и результаты
работы одной не влияют на результаты работы другой) и поэтому
допускающих удобное распараллеливание по инструкциям в чистом виде
*/

// Компиляция:
// gcc -fopenmp main.c -o app

// Запуск
// ./app

#include <stdio.h>
#include <omp.h>

void run_function1()
{
   printf("run_function1\n");
}

void run_function2()
{
   printf("run_function2\n");
}

int main ()  
{
   #pragma omp parallel sections
   {
      #pragma omp section
      run_function1();
      #pragma omp section
      run_function2();
   }
}