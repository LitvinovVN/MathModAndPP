echo "Compiling..."
nvcc kernel.cu -o program.exe
echo "Starting program.exe..."
del program.exp
del program.lib
program.exe