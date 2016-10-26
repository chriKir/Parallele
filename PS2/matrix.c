
void printMatrix(size_t rows, size_t columns, __global float *matrix) {

    for (size_t i = 0; i < rows; ++i) {

        for (size_t j = 0; j < columns; ++j) {
            printf("[%f]\t", matrix[i * columns + j]);
        }
        printf("\n");
    }
}

__kernel void matrix(const __global float *A,
                     const __global float *B,
                     __global float *C,
                     const int l, const int m, const int n
) {

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    float acc = 0.0f;

#ifdef DEBUG
    printf("(%d)(%d) * %d\n", globalRow, globalCol, m);
#endif

    for (int k = 0; k < m; k++) {
        acc += A[globalRow * m + k] * B[k * n + globalCol];
    }

#ifdef DEBUG
    printf("%f\n", acc);
#endif

    C[globalRow * n + globalCol] = acc;
}

