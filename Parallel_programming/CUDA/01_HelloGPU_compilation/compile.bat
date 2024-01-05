echo "!!!!!NOT WORKING!!!!"

echo "Compiling..."
nvcc -O3 kernel.cu -dc -o target.o -gencode arch=compute_52,code=sm_52
nvcc -O3 target.o -o dlink.o -gencode arch=compute_52,code=sm_52 -dlink
nvcc -c main.cpp -o main.o
nvcc dlink.o main.o target.o -o app
echo "Starting app.exe..."
del app.exp
del app.lib
del target.o
del dlink.o
del main.o
app.exe
set /p input="Press Enter to continue..."