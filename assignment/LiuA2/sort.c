#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

void merge(float *a, int n, int m) {
    int i, j, k;
    float *x = malloc(n * sizeof(float));
    for (i = 0, j = m, k = 0; k < n; k++) {
        x[k] = j == n ? a[i++]
             : i == m ? a[j++]
             : a[j] < a[i] ? a[j++]
             : a[i++];
    }
    for (i = 0; i < n; i++) {
        a[i] = x[i];
    }
    free(x);
}

// merge sort
void merge_sort(float *a, int n) {
    if (n < 2)
        return;
    int m = n / 2;
    merge_sort(a, m);
    merge_sort(a + m, n - m);
    merge(a, n, m);
}

int main(int argc, char **argv) {
    int p, id, N;
    // start mpi
    MPI_Init(&argc, &argv);
    // process number
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    // process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (argc != 2) {
        if (id == 0) {
            fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    N = atoi(argv[1]);; // size of the array to sort
    float *data = (float *)malloc(N * sizeof(float)); // array

    srand((unsigned)time(NULL) + id);

    // generate the random numbers on each process
    for (int i = 0; i < N; ++i) {
        data[i] = (float)rand() / RAND_MAX;
    }

    double st_time = MPI_Wtime();

    // sort the numbers on each process
    merge_sort(data, N);
    for (int i = 1; i < N; ++i) {
        if (data[i] < data[i - 1]) {
            printf("Error sort on origin array!\n");
        }
    }

    // printf("%d: ", id);
    // for (int i = 0; i < N; i++)
    // {
    //     printf("%f ", data[i]);
    // }
    // printf("\n");
    

    // calculate the number of elements to send_to/recv_from each process
    int *send_counts = calloc(p, sizeof(int));
    int *send_pos = calloc(p, sizeof(int));
    int *recv_counts = calloc(p, sizeof(int));
    int *recv_pos = calloc(p, sizeof(int));
    float unit_range = 1.0 / p;
    // printf("unit_range: %f\n", unit_range);
    int cur_p = 0;
    for (int i = 0; i < N; ++i) {
        float st_range = cur_p * unit_range;
        float ed_range = (cur_p + 1) * unit_range;
        if (data[i] >= st_range && data[i] < ed_range) {
            ++send_counts[cur_p];
        } else if (data[i] >= ed_range) {
            if (cur_p < (p - 1)) {
                send_pos[cur_p] = i - send_counts[cur_p];
                ++cur_p;
                --i;
            } else {
                ++send_counts[cur_p];
            }
        }
        // printf("%d %d %d: ", id, cur_p, i);
        // for (size_t j = 0; j < p; j++) {
        //     printf("%d ", send_counts[j]);
        // }
        // printf("\n");
    }
    send_pos[p - 1] = N - send_counts[p - 1];

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // for (size_t i = 0; i < p; i++) {
    //     printf("send %d to %d by %d\n", send_counts[i], i, id);
    //     printf("recv %d from %d by %d\n", recv_counts[i], i, id);
    // }
    

    int new_size = 0;
    for (int i = 0; i < p; i++) {
        recv_pos[i] = new_size;
        new_size += recv_counts[i];
    }
    
    float *new_data = (float *)malloc(new_size * sizeof(float)); // new array
    memcpy(new_data + recv_pos[id], data + send_pos[id], recv_counts[id] * sizeof(float));

    MPI_Request *reqs = (MPI_Request *)malloc(2 * p * sizeof(MPI_Request)); // non-blocking communication
    MPI_Status *stats = malloc(2 * p * sizeof(MPI_Status));
    for (int i = 0; i < 2 * p; ++i) {
        reqs[i] = MPI_REQUEST_NULL;
    }
    int req_size = 0;

    for (int i = 0; i < p; i++) {
        if (send_counts[i] != 0)
            MPI_Isend(data + send_pos[i], send_counts[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD, &reqs[req_size++]);
        if (recv_counts[i] != 0)
            MPI_Irecv(new_data + recv_pos[i], recv_counts[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD, &reqs[req_size++]);
    }

    MPI_Waitall(req_size, reqs, stats);
    
    merge_sort(new_data, new_size);

    float cur_st_range = id * unit_range;
    float cur_ed_range = (id + 1) * unit_range;
    for (int i = 0; i < new_size; i++) {
        if (new_data[i] < new_data[i - 1]) {
            printf("Error sort on origin array!\n");
        }
        if (new_data[i] < cur_st_range || (id < (p - 1) && new_data[i] >= cur_ed_range) || new_data[i] > cur_ed_range) {
            printf("Number in error range!\n");
        }
    }

    // printf("%d: ", id);
    // for (int i = 0; i < new_size; i++)
    // {
    //     printf("%f ", new_data[i]);
    // }
    // printf("\n");

    double ed_time = MPI_Wtime();
    printf("Process %d finished in %f seconds.\n", id, ed_time - st_time);

    // free memory
    free(data);
    free(reqs);
    free(stats);
    free(send_counts);
    free(recv_counts);
    free(send_pos);
    free(recv_pos);
    free(new_data);

    MPI_Finalize();
    return 0;
}
