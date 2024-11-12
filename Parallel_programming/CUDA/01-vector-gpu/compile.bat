echo "Compiling..."
nvcc main.cu -o app.exe -gencode arch=compute_75,code=sm_75
echo "Starting app.exe..."
del app.exp
del app.lib
app.exe
set /p input="Press Enter to continue..."