// before running, need this command: 
// ulimit -s unlimited

#include "stdio.h"
int main()
{
    int DIM=4*2048;
    int matrix[DIM*DIM];
    int n,m;

    for (n=0;n<DIM;n++){
        for(m=0;m<DIM;m++){
            matrix[n*DIM+m]=n+m; //initialize 
        } 
    }
    int odds=0;
#pragma omp parallel for collapse(2) reduction(+:odds)
    for (n=0;n<DIM;n++){
        for(m=0;m<DIM;m++){
                if(matrix[n*DIM+m]%2!=0) odds++;
        }
    }
    printf(" odds %d \n",odds);
}
