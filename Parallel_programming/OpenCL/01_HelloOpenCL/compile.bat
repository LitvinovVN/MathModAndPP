echo "Compiling..."
rustc main.rs -o app.exe
echo "Starting app.exe..."
del app.pdb
app.exe
set /p input="Press Enter to continue..."