void Build_derived_type(
         float*         a_ptr           /* in   */,
         float*         b_ptr           /* in   */,
         int*           n_ptr           /* in   */,
         MPI_Datatype*  mesg_mpi_t_ptr  /* out  */) {

    int block_lengths[3];
    MPI_Aint displacements[3];
    MPI_Datatype typelist[3];

    MPI_Aint start_address;
    MPI_Aint address;
    block_lengths[0]=block_lengths[1]=block_lengths[2]=1;

    typelist[0] = MPI_FLOAT;
    typelist[1] = MPI_FLOAT;
    typelist[2] = MPI_INT;
    /* First element, a, is at displacement 0      */
    displacements[0] = 0;
    /* Calculate displacements relative to a */
    MPI_Get_address(a_ptr, &start_address);

    MPI_Get_address(b_ptr, &address);
    displacements[1] = address - start_address;

    MPI_Get_address(n_ptr, &address);
    displacements[2] = address - start_address;

    MPI_Type_create_struct(3, block_lengths, displacements,
        typelist, mesg_mpi_t_ptr);
    MPI_Type_commit(mesg_mpi_t_ptr);}         }

void Get_data3(
         float*  a_ptr    /* out */,
         float*  b_ptr    /* out */,
         int*    n_ptr    /* out */,
         int     my_rank  /* in  */) {
    MPI_Datatype  mesg_mpi_t; /* MPI type corresponding */
                              /* to 2 floats and an int */

    if (my_rank == 0){
        printf("Enter a, b, and n\n");
        scanf("%f %f %d", a_ptr, b_ptr, n_ptr);
    }

    Build_derived_type(a_ptr, b_ptr, n_ptr, &mesg_mpi_t);
    MPI_Bcast(a_ptr, 1, mesg_mpi_t, 0, MPI_COMM_WORLD);
}