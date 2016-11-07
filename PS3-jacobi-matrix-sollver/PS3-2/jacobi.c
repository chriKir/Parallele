
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
 * @param goal column for x
 * @param matrix_size size of A
 */
__kernel void jacobi(const __global float *A,
                     const __global float *b,
                     __global float *x,
                     const unsigned int matrix_size,
                     const unsigned int iterations
) {
    int i = get_global_id(0);

    int iteration = 0;

    do {

        float x_new = 0.0;
        for (int j = 0; j < matrix_size; j++) {
            if (i != j)
                x_new += A[i * matrix_size + j] * x[j];
        }
        x_new = (b[i] - x_new) / A[i * matrix_size + i];

        x[i] = x_new;

        iteration++;

    } while (iteration < iterations);

#ifdef DEBUG
    printMatrix(matrix_size, 1, x);
#endif
}

