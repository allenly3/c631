// for this program to work, the number of threads specified at compile time and run time
// must match (here 4 in each case).
// compile with
// gcc -DNTHREADS=4 -fopenmp countodds_critical.c
// and then run with
// ulimit -s unlimited
// OMP_NUM_THREADS=4 ./a.out

#include<stdio.h>
#include "stdlib.h"

#include <sys/time.h>

int main()
{

struct timeval  tv1, tv2;
int numthreads=NTHREADS;
//int result[numthreads];
int DIM=4*2048;
int matrix[DIM*DIM];

int n,m;

for (n=0;n<DIM;n++){
    for(m=0;m<DIM;m++){
        matrix[n*DIM+m]=n+m; //initialize
    }
}



int odds=0;

gettimeofday(&tv1, NULL);
#pragma omp parallel
{
int ID = omp_get_thread_num();
//int p=omp_get_num_threads();
//printf("hello(%d) out of %d \n",ID,p);
//if(numthreads!=p) printf("incorrect number of threads!!! \n");

int i,j;
int chunksize=DIM/numthreads;
int mystart=ID*chunksize;
int myend=mystart+chunksize;

int result_local=0;
for (i=mystart;i<myend;i++){
    for(j=0;j<DIM;j++){
        if(matrix[i*DIM+j]%2!=0)  result_local++;

    }
}
//printf("%d %d %d result %d\n",chunksize,mystart,myend,result[ID]);

#pragma omp critical
{
odds=odds+result_local;
}

}
gettimeofday(&tv2, NULL);

printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec));


printf(" odds %d \n",odds);



}

