/*
    CMD:
    gcc -o q1p1  q1p1.c -lm
    ./q1p1
*/

#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define N 100  // 100 particles 


/**
    x: [N][0]
    y: [N][1]
*/
double computeDistance(double particles[N][2], int i, int j) {
    double dx = particles[i][0] - particles[j][0];
    double dy = particles[i][1] - particles[j][1];
    return sqrt(dx * dx + dy * dy);
}

double findMinimumDistance(double particles[N][2]) {
    double minDistance = DBL_MAX;
    int i , j ; 
    for ( i = 0; i < N; i++) {
        for ( j = i + 1; j < N; j++) {
            double distance = computeDistance(particles, i, j);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
    }
    return minDistance;
}

int main() {
    double particles[N][2];
    srand(time(NULL));
    int i; 
    for ( i = 0; i < N; ++i) {
        particles[i][0] = (double)rand() / RAND_MAX;
        particles[i][1] = (double)rand() / RAND_MAX;
    }

    double minDistance = findMinimumDistance(particles);

    printf("Minimum distance between particles: %lf\n", minDistance);

    return 0;
}