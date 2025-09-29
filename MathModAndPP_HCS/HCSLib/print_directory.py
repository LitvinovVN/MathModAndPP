import os
import sys
import win32print
import win32ui
from PIL import Image
import tempfile

def print_file(file_path, printer_name=None):
    """
    Отправляет файл на печать в зависимости от его типа
    """
    try:
        # Определяем тип файла по расширению
        ext = os.path.splitext(file_path)[1].lower()
        
        if printer_name is None:
            printer_name = win32print.GetDefaultPrinter()
        
        print(f"Печатаем: {file_path} на принтере: {printer_name}")
        
        if ext in ['.txt', '.h', '.cpp', '.hpp', '.c', '.cu', '.css', '.xml', '.json']:
            # Текстовые файлы
            _print_text_file(file_path, printer_name)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
            # Изображения
            _print_image_file(file_path, printer_name)
        elif ext in ['.pdf']:
            # PDF файлы (требует установки Ghostscript)
            _print_pdf_file(file_path, printer_name)
        else:
            # Для остальных файлов пытаемся открыть через ассоциированную программу
            _print_through_shell(file_path, printer_name)
            
    except Exception as e:
        print(f"Ошибка при печати файла {file_path}: {e}")

def _print_text_file(file_path, printer_name):
    """Печать текстового файла"""
    try:
        # Открываем принтер
        hprinter = win32print.OpenPrinter(printer_name)
        try:
            # Начинаем работу с документом
            hjob = win32print.StartDocPrinter(hprinter, 1, (os.path.basename(file_path), None, "RAW"))
            try:
                win32print.StartPagePrinter(hprinter)
                
                # Читаем и отправляем содержимое файла
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Преобразуем в bytes и отправляем на принтер
                    win32print.WritePrinter(hprinter, content.encode('utf-8', errors='ignore'))
                
                win32print.EndPagePrinter(hprinter)
            finally:
                win32print.EndDocPrinter(hprinter)
        finally:
            win32print.ClosePrinter(hprinter)
    except Exception as e:
        print(f"Ошибка печати текстового файла: {e}")

def _print_image_file(file_path, printer_name):
    """Печать изображения"""
    try:
        # Открываем изображение с помощью PIL
        image = Image.open(file_path)
        
        # Создаем контекст устройства для принтера
        hprinter = win32print.OpenPrinter(printer_name)
        try:
            # Получаем характеристики принтера
            printer_info = win32print.GetPrinter(hprinter, 2)
            dc = win32ui.CreateDC()
            dc.CreatePrinterDC(printer_name)
            
            # Начинаем документ
            dc.StartDoc(os.path.basename(file_path))
            dc.StartPage()
            
            # Масштабируем изображение под размер страницы
            page_width = dc.GetDeviceCaps(110)  # HORZRES
            page_height = dc.GetDeviceCaps(111)  # VERTRES
            
            img_width, img_height = image.size
            
            # Рассчитываем соотношение сторон
            ratio = min(page_width / img_width, page_height / img_height) * 0.9
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Позиционируем по центру
            x = (page_width - new_width) // 2
            y = (page_height - new_height) // 2
            
            # Конвертируем изображение в bitmap и печатаем
            temp_bmp = tempfile.NamedTemporaryFile(suffix='.bmp', delete=False)
            image.save(temp_bmp.name, 'BMP')
            temp_bmp.close()
            
            # Создаем bitmap объект
            bmp = win32ui.CreateBitmap()
            bmp.LoadBitmapFile(temp_bmp.name)
            
            # Создаем совместимый DC и выбираем bitmap
            mem_dc = dc.CreateCompatibleDC()
            mem_dc.SelectObject(bmp)
            
            # Копируем bitmap на принтер
            dc.StretchBlt((x, y, x + new_width, y + new_height), 
                         mem_dc, (0, 0, img_width, img_height), 
                         win32ui.SRCCOPY)
            
            # Завершаем документ
            dc.EndPage()
            dc.EndDoc()
            
            # Удаляем временный файл
            os.unlink(temp_bmp.name)
            
        finally:
            win32print.ClosePrinter(hprinter)
    except Exception as e:
        print(f"Ошибка печати изображения: {e}")

def _print_pdf_file(file_path, printer_name):
    """Печать PDF файла (упрощенная версия)"""
    try:
        # Для PDF файлов используем системную печать через shell
        os.startfile(file_path, "print")
        print(f"PDF файл {file_path} отправлен в очередь печати")
    except Exception as e:
        print(f"Ошибка печати PDF: {e}. Убедитесь, что есть программа для просмотра PDF.")

def _print_through_shell(file_path, printer_name):
    """Печать через системную ассоциацию файлов"""
    try:
        os.startfile(file_path, "print")
        print(f"Файл {file_path} отправлен в очередь печати через ассоциированную программу")
    except Exception as e:
        print(f"Не удалось напечатать файл {file_path}: {e}")

def list_printers():
    """Выводит список доступных принтеров"""
    printers = win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL | 
                                      win32print.PRINTER_ENUM_CONNECTIONS)
    print("Доступные принтеры:")
    for i, printer in enumerate(printers):
        print(f"{i + 1}. {printer[2]}")

def print_directory(directory_path, printer_name=None, file_extensions=None):
    """
    Рекурсивно печатает все файлы в указанной директории
    
    Args:
        directory_path: Путь к директории
        printer_name: Имя принтера (если None, используется принтер по умолчанию)
        file_extensions: Список расширений для фильтрации (если None, печатаются все файлы)
    """
    if not os.path.exists(directory_path):
        print(f"Ошибка: Директория {directory_path} не существует")
        return
    
    if printer_name is None:
        printer_name = win32print.GetDefaultPrinter()
    
    print(f"Начинаем печать файлов из директории: {directory_path}")
    print(f"Используемый принтер: {printer_name}")
    
    file_count = 0
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Фильтрация по расширению
            if file_extensions:
                ext = os.path.splitext(file)[1].lower()
                if ext not in file_extensions:
                    continue
            
            print_file(file_path, printer_name)
            file_count += 1
    
    print(f"Обработка завершена. Всего отправлено на печать: {file_count} файлов")

def main():
    """Основная функция"""
    if len(sys.argv) < 2:
        print("Использование: python print_directory.py <путь_к_директории> [имя_принтера]")
        print("Пример: python print_directory.py C:\\MyFolder")
        print("Пример: python print_directory.py C:\\MyFolder \"My Printer\"")
        print("\nДоступные принтеры:")
        list_printers()
        return
    
    directory_path = sys.argv[1]
    printer_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Опционально: можно указать конкретные расширения файлов для печати
    # file_extensions = ['.txt', '.pdf', '.jpg', '.png']  # раскомментируйте для фильтрации
    file_extensions = None  # печатать все файлы
    
    # Подтверждение перед началом печати
    response = input(f"Вы уверены, что хотите напечатать все файлы из {directory_path}? (y/n): ")
    if response.lower() != 'y':
        print("Операция отменена")
        return
    
    print_directory(directory_path, printer_name, file_extensions)

if __name__ == "__main__":
    # Проверяем, что скрипт запущен на Windows
    if os.name != 'nt':
        print("Этот скрипт работает только на Windows")
        sys.exit(1)
    
    # Проверяем наличие необходимых библиотек
    try:
        import win32print
        import win32ui
        from PIL import Image
    except ImportError as e:
        print(f"Ошибка: Не удалось импортировать необходимые библиотеки: {e}")
        print("Установите необходимые пакеты:")
        print("pip install pywin32 Pillow")
        sys.exit(1)
    
    main()