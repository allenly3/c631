/*
histogram program illustrating the use of locks

example data used will be uniformly distributed among the histogram bins

*/

#include "omp.h"
#include "stdio.h"

#define NBINS 10

int sample(int v){
    int ibin;
    ibin= v % NBINS;
    return ibin; // associates each value of v with different histogram bin

}

int main(){

int N=100;
int arr[N];  //data
int histogram[NBINS];
int i;
omp_lock_t histogram_locks[NBINS];

for(i=0;i<N;i++){
    arr[i]=1.0*i; 
}

for(i=0;i<NBINS;i++){
    histogram[i]=0;
}

for(i=0;i<NBINS;i++){
    omp_init_lock(&histogram_locks[i]);
}

#pragma omp parallel for
for(i=0;i<N;i++){
    int ival; //this variable must be declared inside the loop to be private for each thread
    ival = sample(arr[i]);
//    printf("%d %d \n",ival,arr[i]);
    omp_set_lock(&histogram_locks[ival]);
    histogram[ival]++;
    omp_unset_lock(&histogram_locks[ival]);
}

for(i=0;i<NBINS;i++)
omp_destroy_lock(&histogram_locks[i]);

for(i=0;i<NBINS;i++){
    printf("histogram bin=%d count=%d \n",i,histogram[i]); 
}

return 0;
}
