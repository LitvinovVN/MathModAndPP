echo "Compiling..."
nvcc kernel.cu -o app.exe
echo "Starting app.exe..."
del app.exp
del app.lib
app.exe
set /p input="Press Enter to continue..."