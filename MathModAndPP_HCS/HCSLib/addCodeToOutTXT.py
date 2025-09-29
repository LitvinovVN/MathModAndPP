import os
from pathlib import Path

def collect_c_cpp_files():
    """
    Простая версия без аргументов - настройте пути ниже
    """
    # Настройте эти пути под ваши needs
    root_directory = "."  # Текущая директория, или укажите полный путь
    output_file = "out.txt"
    
    extensions = {'.c', '.cu', '.h', '.cpp', '.hpp'}
    file_count = 0
    
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("C/C++ Files Collection\n")
            out_f.write("=" * 50 + "\n\n")
            
            for root, dirs, files in os.walk(root_directory):
                # Пропускаем скрытые папки
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    file_path = Path(root) / file
                    file_ext = file_path.suffix.lower()
                    
                    if file_ext in extensions:
                        file_count += 1
                        
                        # Записываем информацию о файле
                        out_f.write(f"FILE: {file}\n")
                        out_f.write(f"PATH: {file_path}\n")
                        out_f.write(f"EXTENSION: {file_ext}\n")
                        out_f.write(f"SIZE: {file_path.stat().st_size} bytes\n")
                        out_f.write("-" * 40 + "\n")
                        
                        # Читаем содержимое
                        try:
                            with open(file_path, 'r', encoding='utf-8') as in_f:
                                content = in_f.read()
                                out_f.write("CONTENT:\n")
                                out_f.write(content)
                        except Exception as e:
                            out_f.write(f"ERROR READING: {e}\n")
                        
                        out_f.write("\n" + "=" * 50 + "\n\n")
            
            out_f.write(f"TOTAL FILES PROCESSED: {file_count}\n")
        
        print(f"Done! Found {file_count} files")
        print(f"Output: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    collect_c_cpp_files()