int dim_sizes[3];  int wrap_around[3];

for(i=0;i<3;i++){ 

dim_sizes[i] = q; wrap_around[i] = 0;

}

int MPI_Cart_create(MPI_COMM_WORLD, 3, dim_sizes, wrap_around , 1 , &comm_cart)