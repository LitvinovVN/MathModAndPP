import numpy as np

def read_data_file(filename):
    """
    Читает файл с данными и возвращает матрицу и размеры ячеек
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            # Читаем первую строку с размерами матрицы
            first_line = file.readline().strip()
            x_size, y_size = map(int, first_line.split())
            
            # Читаем вторую строку с размерами ячеек
            second_line = file.readline().strip()
            cell_x, cell_y = map(int, second_line.split())
            
            # Создаем матрицу, заполненную нулями
            matrix = np.zeros((y_size, x_size))
            
            # Читаем остальные строки с данными
            for line in file:
                line = line.strip()
                if line:  # Пропускаем пустые строки
                    parts = line.split()
                    if len(parts) >= 3:
                        x_ind, y_ind, val = map(float, parts[:3])
                        matrix[int(y_ind), int(x_ind)] = val
            
            return matrix, x_size, y_size, cell_x, cell_y
            
    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден")
        return None, 0, 0, 0, 0
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None, 0, 0, 0, 0

def create_compact_heatmap_svg(matrix, cell_x, cell_y, output_filename, title="Тепловая карта"):
    """
    Создает компактную тепловую карту с градиентом от белого к черному
    """
    height, width = matrix.shape
    
    # Увеличиваем высоту SVG, чтобы подписи легенды не обрезались
    svg_width = width * cell_x + 60
    svg_height = height * cell_y + 45
    
    # Находим минимальное и максимальное значение
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    if max_val == min_val:
        max_val = min_val + 1
    
    # Создаем улучшенное SVG содержимое
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
    <title>{title}</title>
    <style>
        .cell {{ stroke: #666; stroke-width: 0.3; }}
        .micro-label {{ font-family: Arial, sans-serif; font-size: 7px; fill: #444; }}
        .small-label {{ font-family: Arial, sans-serif; font-size: 8px; fill: #333; }}
        .compact-title {{ font-family: Arial, sans-serif; font-size: 9px; font-weight: bold; fill: #222; }}
        .tiny-value {{ font-family: Arial, sans-serif; font-size: 7px; fill: #fff; }}
        .legend-label {{ font-family: Arial, sans-serif; font-size: 6px; fill: #333; }}
    </style>
    
    <!-- Компактный заголовок с уменьшенным отступом -->
    <text x="{svg_width/2}" y="10" text-anchor="middle" class="compact-title">{title}</text>
    
    <!-- Тепловая карта с уменьшенным отступом -->
    <g transform="translate(30, 20)">
'''
    
    # Функция для получения цвета от белого к черному
    def get_color(value):
        normalized = (value - min_val) / (max_val - min_val)
        
        # Градиент от белого к черному
        intensity = int((1 - normalized) * 255)  # 1-normalized чтобы белый = min, черный = max
        return f"rgb({intensity}, {intensity}, {intensity})"
    
    # Создаем ячейки тепловой карты
    for y in range(height):
        for x in range(width):
            value = matrix[y, x]
            color = get_color(value)
            
            x_pos = x * cell_x
            y_pos = y * cell_y
            
            # Рисуем ячейку
            svg_content += f'        <rect x="{x_pos}" y="{y_pos}" width="{cell_x}" height="{cell_y}" fill="{color}" class="cell"/>\n'
            
            # Добавляем текст значения только для достаточно больших ячеек и ненулевых значений
            if cell_x > 25 and cell_y > 20 and value != 0:
                # Для черно-белой карты используем контрастные цвета текста
                text_color = "white" if value > (max_val + min_val) / 2 else "black"
                text_x = x_pos + cell_x / 2
                text_y = y_pos + cell_y / 2
                
                # Форматируем значение для компактности
                if abs(value) >= 100:
                    value_text = f"{value:.0f}"
                elif abs(value) >= 10:
                    value_text = f"{value:.1f}"
                else:
                    value_text = f"{value:.2f}"
                
                svg_content += f'        <text x="{text_x}" y="{text_y}" text-anchor="middle" dominant-baseline="middle" class="tiny-value" fill="{text_color}">{value_text}</text>\n'
    
    svg_content += '    </g>\n'
    
    # КОМПАКТНАЯ ЛЕГЕНДА - градиент от белого к черному
    legend_width = min(75, svg_width - 60)
    legend_x = 30
    legend_y = height * cell_y + 30
    
    # Рассчитываем промежуточные значения для легенды
    mid_val = (min_val + max_val) / 2
    
    svg_content += f'''
    <!-- КОМПАКТНАЯ ЛЕГЕНДА (от белого к черному) -->
    <g transform="translate({legend_x}, {legend_y})">
        <!-- Градиентная полоса с рамкой -->
        <defs>
            <linearGradient id="bwGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:rgb(255,255,255);stop-opacity:1" />
                <stop offset="100%" style="stop-color:rgb(0,0,0);stop-opacity:1" />
            </linearGradient>
        </defs>
        
        <!-- Основная градиентная полоса -->
        <rect x="0" y="0" width="{legend_width}" height="4" fill="url(#bwGradient)" stroke="#666" stroke-width="0.2"/>
        
        <!-- Деления и подписи -->
        <line x1="0" y1="4" x2="0" y2="6" stroke="#666" stroke-width="0.3"/>
        <line x1="{legend_width/2}" y1="4" x2="{legend_width/2}" y2="7" stroke="#666" stroke-width="0.3"/>
        <line x1="{legend_width}" y1="4" x2="{legend_width}" y2="6" stroke="#666" stroke-width="0.3"/>
        
        <!-- Подписи значений с увеличенными отступами -->
        <text x="0" y="15" text-anchor="start" class="legend-label">{min_val:.1f}</text>
        <text x="{legend_width/2}" y="15" text-anchor="middle" class="legend-label">{mid_val:.1f}</text>
        <text x="{legend_width}" y="15" text-anchor="end" class="legend-label">{max_val:.1f}</text>
    </g>
</svg>'''
    
    # Сохраняем SVG файл
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"Тепловая карта с черно-белым градиентом сохранена в файл: {output_filename}")
    print(f"Размеры: {svg_width} × {svg_height} пикселей")
    print(f"Диапазон значений: {min_val:.3f} - {max_val:.3f}")
    print(f"Цветовая схема: белый ({min_val:.1f}) → черный ({max_val:.1f})")

def main():
    input_filename = "1.txt"
    output_filename = "1.svg"
    
    # Читаем данные из файла
    matrix, x_size, y_size, cell_x, cell_y = read_data_file(input_filename)
    
    if matrix is not None:
        print(f"Размер матрицы: {y_size} × {x_size}")
        print(f"Размер ячейки: {cell_x} × {cell_y} пикселей")
        print(f"Ненулевых элементов: {np.count_nonzero(matrix)}")
        
        # Создаем тепловую карту с черно-белым градиентом
        create_compact_heatmap_svg(matrix, cell_x, cell_y, output_filename)
        
        # Дополнительная статистика
        print(f"\nСтатистика данных:")
        print(f"Минимум: {np.min(matrix):.3f}")
        print(f"Максимум: {np.max(matrix):.3f}")
        print(f"Среднее: {np.mean(matrix):.3f}")
        
    else:
        print("Не удалось прочитать данные из файла")

if __name__ == "__main__":
    main()