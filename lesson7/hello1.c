#include "stdio.h"
#include "stdlib.h"
int main()
{

#pragma omp parallel
{
int ID = omp_get_thread_num();
int p=omp_get_num_threads();
printf("hello(%d) out of %d \n",ID,p);
printf("how(%d) out of %d \n",ID,p);
printf("are(%d) out of %d \n",ID,p);
printf("you(%d) out of %d \n",ID,p);



}


}

