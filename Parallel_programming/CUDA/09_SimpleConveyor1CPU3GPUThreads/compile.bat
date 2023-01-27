echo "Compiling..."
nvcc main.cu -o app
echo "Starting app.exe..."
del app.exp
del app.lib
app.exe
set /p input="Press Enter to continue..."