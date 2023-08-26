# Решение одномерного уравнения теплопроводности методом прогонки

$$ \rho c \frac{\partial T}{\partial t}
 = \lambda 
 \frac{\partial^2 T}{\partial x^2},
 0<x<L $$

Начальные условия:
$$ t=0: T=T_0, 0 \le x \le L $$

Граничные условия:
$$ x=0: T=T_l, t \gt 0 $$
$$ x=L: T=T_r, t \gt 0 $$

## Запуск
g++ main.cpp -o app

./app

python3 plot.py