#!/usr/bin/env python3
"""
Визуализация результатов вычисления градиента
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_results():
    # Чтение результатов
    df = pd.read_csv('gradient_results.csv')
    
    # Определение размеров сетки
    x_unique = sorted(df['x'].unique())
    y_unique = sorted(df['y'].unique())
    nx = len(x_unique)
    ny = len(y_unique)
    
    # Преобразование в матрицы
    X = df['x'].values.reshape(nx, ny)
    Y = df['y'].values.reshape(nx, ny)
    F = df['f'].values.reshape(nx, ny)
    GX = df['grad_x'].values.reshape(nx, ny)
    GY = df['grad_y'].values.reshape(nx, ny)
    GM = df['grad_magnitude'].values.reshape(nx, ny)
    
    # Создание графиков
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Исходная функция
    ax1 = fig.add_subplot(231, projection='3d')
    surf1 = ax1.plot_surface(X, Y, F, cmap='viridis', alpha=0.8)
    ax1.set_title('Функция f(x,y)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x,y)')
    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 2. Компонента X градиента
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(X, Y, GX, cmap='plasma', alpha=0.8)
    ax2.set_title('Градиент ∂f/∂x')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('∂f/∂x')
    plt.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 3. Компонента Y градиента
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(X, Y, GY, cmap='plasma', alpha=0.8)
    ax3.set_title('Градиент ∂f/∂y')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('∂f/∂y')
    plt.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # 4. Модуль градиента
    ax4 = fig.add_subplot(234, projection='3d')
    surf4 = ax4.plot_surface(X, Y, GM, cmap='hot', alpha=0.8)
    ax4.set_title('Модуль градиента |∇f|')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('|∇f|')
    plt.colorbar(surf4, ax=ax4, shrink=0.5)
    
    # 5. Векторное поле градиента
    ax5 = fig.add_subplot(235)
    # Прореживание векторов для лучшей визуализации
    stride = max(1, nx // 20)
    skip = (slice(None, None, stride), slice(None, None, stride))
    quiver = ax5.quiver(X[skip], Y[skip], GX[skip], GY[skip], GM[skip], 
                       cmap='jet', scale=50, width=0.002)
    ax5.set_title('Векторное поле градиента')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    plt.colorbar(quiver, ax=ax5, shrink=0.8)
    
    # 6. Контурный график с градиентом
    ax6 = fig.add_subplot(236)
    contour = ax6.contourf(X, Y, F, levels=20, cmap='viridis')
    ax6.quiver(X[skip], Y[skip], GX[skip], GY[skip], 
              color='white', scale=50, width=0.003)
    ax6.set_title('Контуры функции и градиент')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    plt.colorbar(contour, ax=ax6, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('gradient_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize_results()