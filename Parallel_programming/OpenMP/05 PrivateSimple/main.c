/*
   5. В каждую ячейку одномерного массива нужно записать
   индекс этой ячейки, возведенный в 4-ую степень.
   Элемент private в директиве pragma задает
   через запятую перечень локальных (приватных) для
   каждого потока переменных.
   В данном случае такая переменная одна: tmp.
*/

// Компиляция:
// gcc -std=c99 -fopenmp main.c -o app

// Запуск
// ./app

#include <stdio.h>
#include <omp.h>

#define N 10

int main ()  
{
   int* a[N];
   for(int i=0;i<N;i++) a[i] = i;

   int i, tmp;
   #pragma omp parallel for private(tmp)
   for (i = 0; i < N; ++i)
   {
      tmp = i*i;
      a[i] = tmp*tmp;
   }

   for(int i=0;i<N;i++)
      printf("%d: %d\n", i, a[i]);
}