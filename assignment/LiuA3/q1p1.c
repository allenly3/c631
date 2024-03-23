/*
    CMD:
    gcc -o q1p1  q1p1.c
    ./q1p1
*/

#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define N 10 

double computeDistance(double particles[N][2], int i, int j) {
    double dx = particles[i][0] - particles[j][0];
    double dy = particles[i][1] - particles[j][1];
    return sqrt(dx * dx + dy * dy);
}

double findMinimumDistance(double particles[N][2]) {
    double minDistance = DBL_MAX;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double distance = computeDistance(particles, i, j);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
    }
    return minDistance;
}

int main() {
    // 使用随机数生成粒子的坐标
    double particles[N][2];
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        particles[i][0] = (double)rand() / RAND_MAX;
        particles[i][1] = (double)rand() / RAND_MAX;
    }

    // 找到最小距离
    double minDistance = findMinimumDistance(particles);

    // 输出结果
    printf("Minimum distance between particles: %lf\n", minDistance);

    return 0;
}