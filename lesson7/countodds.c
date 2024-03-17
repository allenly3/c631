
#include "stdio.h"
#include "omp.h"
int main()
{
    //np = omp_get_num_threads();
    //id = omp_get_thread_num();
    int numthreads=NTHREADS;
    int result[numthreads];
    int DIM=16*16; //int DIM=4*2048;
    int matrix[DIM*DIM];
    int n,m;

    for (n=0;n<DIM;n++){
        for(m=0;m<DIM;m++){
            matrix[n*DIM+m]=n+m; //initialize
        }
    }

#pragma omp parallel
    {
        int ID = omp_get_thread_num();

        int i,j;
        int chunksize=DIM/numthreads;
        int mystart=ID*chunksize;
        int myend=mystart+chunksize;

        result[ID]=0;
        for (i=mystart;i<myend;i++){
            for(j=0;j<DIM;j++){
                if(matrix[i*DIM+j]%2!=0) result[ID]++;
            }
        }
    } // end of #pragma

    int odds=0;
    int k;
    for (k=0;k<numthreads;k++) odds=odds+result[k];

    printf(" odds %d \n",odds);
}