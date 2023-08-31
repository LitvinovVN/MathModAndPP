# Выводит количество процессов и идентификатор текущего процесса
# Запуск: mpiexec -n 4 python 01.py

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

print(f'Hello from process {rank} of {numprocs}')