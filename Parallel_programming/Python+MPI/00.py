# Последовательный алгоритм умножения матрицы на вектор
# python 00.py
#             i = 1   i = 2   i = 3    ...    i = N
#   j = 1     A[1,1]    x       x      ...      x
#   j = 2       x       x       x      ...      x
#   j = 3       x     A[j,i]    x      ...      x
#    ...
#   j = M       x       x       x      ...    A[M,N]
#

from numpy import empty

f1 = open('in.dat', 'r')

# Количество строк
M = int(f1.readline())
# Количество столбцов
N = int(f1.readline())

f1.close()
print(f"Размерности: M = {M}, N = {N}")

A = empty((M, N))
x = empty(N)
b = empty(M)

print('AData.dat')
f2 = open('AData.dat', 'r')
for j in range(M):
    for i in range(N):
        A[j, i] = float(f2.readline())
        print(f'{A[j, i]} ', end = ' ')
    print()
f2.close()

print('xData.dat')
f3 = open('xData.dat', 'r')
for i in range(N):
    x[i] = float(f3.readline())
    print(f'{x[i]} ', end = ' ')
f3.close()

for j in range(M):
    b[j] = 0
    for i in range(N):
        b[j] += A[j, i]*x[i]

print('\nResults.dat')
f4 = open('Results.dat', 'w')
for j in range(M):
    print(b[j], file=f4)
    print(f'{b[j]} ', end = ' ')
f4.close()