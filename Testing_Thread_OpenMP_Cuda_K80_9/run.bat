@echo off
echo "Compiling..."
g++ main.cpp -o app.exe
echo "Starting app.exe..."
app.exe
set /p input="Press Enter to continue..."