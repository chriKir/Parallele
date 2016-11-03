
//#define DEBUG

void printMatrix(size_t rows, size_t columns, __global float *matrix) {

    for (size_t i = 0; i < rows; ++i) {

        for (size_t j = 0; j < columns; ++j) {
            printf("[%f]\t", matrix[i * columns + j]);
        }
        printf("\n");
    }
}


/**
 * solves Ax = b approximately
 * @param A Matrix A
 * @param b Column b
 * @param x1 start column for x
 * @param x2 goal column for x
 * @param matrix_size size of A
 */
__kernel void jacobi(const __global float *A,
                     const __global float *b,
                     __global float *x1,
                     __global float *x2,
                     const unsigned int matrix_size
) {
    int i = get_global_id(0);

    x2[i] = 0.0;
    for (int j = 0; j < matrix_size; j++) {
        if (i != j)
            x2[i] += A[i * matrix_size + j] * x1[j];
    }
    x2[i] = (b[i] - x2[i]) / A[i * matrix_size + i];

#ifdef DEBUG
    printMatrix(matrix_size, 1, x2);
#endif
}

