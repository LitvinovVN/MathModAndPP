echo "Compiling..."
nvcc kernel.cu -o app.exe
echo "Starting program.exe..."
del app.exp
del app.lib
app.exe