//
// Created by Roland Gritzer on 13.10.16.
//

#include <iostream>
#include <cmath>
#include <malloc.h>

#include "ClLoader.h"

#include "time_ms.h"


#define VALUE int
#define N 100

VALUE vec[N];

// Fill array with: { 1, 2, 3, ..., N}
void fillArray(VALUE *vec);

void printArray(VALUE *vec);

// Gauß'sche Summenformel: 1+2+3+...+N == (1/2) * N * (N+1)
void validateSum(int sum);

int main() {

    printf("Filling array of length %d and holding INT values\n", N);
    fillArray(vec);
//	printArray(vec);

    unsigned long start_time = time_ms();
    VALUE accumulator = 0;
    for (int i = 0; i < N; ++i) {
        accumulator += vec[i];
    }

    // print total time
    printf("Total Time: %9lu ms\n", time_ms() - start_time);
    validateSum(accumulator);

    return 0;
}


void printArray(VALUE *vec) {
    int i = 0;
    printf("Array: { ");
    for (; i < N - 1; ++i) {
        printf("%d, ", vec[i]);
    }
    printf("%d}\n", vec[i]);
}


void fillArray(VALUE *vec) {
    for (int i = 1; i <= N; ++i) {
        vec[i - 1] = (VALUE) i;
    }
}

void validateSum(int sum) {
    VALUE check = N * (N + 1);
    if (sum == check / 2) {
        printf("validation correct: %d == %d\n", check / 2, sum);
    } else {
        printf("validation fail: %d != %d\n", check / 2, sum);
    }
}
