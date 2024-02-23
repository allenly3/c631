To run MPI:

mpicc -o NAME.x NAME.c

mpirun -np 8 ./NAME.x

-------------------------------------------------------

To run openMP:

gcc -fopenmp -O2 test.c -o test.x

OMP_NUM_THREADS=4 ./test.x