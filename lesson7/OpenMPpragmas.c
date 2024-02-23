#include "stdio.h"
#include "stdlib.h"
int main()
{

    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        int p=omp_get_num_threads();
        printf("hello(%d) out of %d \n",ID,p);
    }

}
