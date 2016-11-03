//
// Created by Roland Gritzer on 13.10.16.
//

#include <iostream>
#include <malloc.h>

// TODO: type..
//#define TYPE float

#include "ClLoader.h"

//#define PRINT_MTX


#define ITERATIONS 1000

void printMatrix(int rows, int columns, cl_float *matrix) {

    for (int i = 0; i < rows; ++i) {
        std::string line = "";

        for (int j = 0; j < columns; ++j) {
            line.append("[");
            line.append(std::to_string(matrix[i * columns + j]));
            line.append("]\t");
        }
        std::cout << line << "\n";
    }
}

void init_diag_dom_near_identity_matrix(int Ndim, cl_float *A) {

    int i, j;
    cl_float sum;

//
// Create a random, diagonally dominant matrix.  For
// a diagonally dominant matrix, the diagonal element
// of each row is great than the sum of the other
// elements in the row.  Then scale the matrix so the
// result is near the identiy matrix.
    for (i = 0; i < Ndim; i++) {
        sum = (cl_float) 0.0;
        for (j = 0; j < Ndim; j++) {
            *(A + i * Ndim + j) = (std::rand() % 23) / (cl_float) 1000.0;
            sum += *(A + i * Ndim + j);
        }
        *(A + i * Ndim + i) += sum;

        // scale the row so the final matrix is almost an identity matrix;wq
        for (j = 0; j < Ndim; j++)
            *(A + i * Ndim + j) /= sum;
    }

}

int main() {

    try {

        ClLoader *loader = new ClLoader("../jacobi.c", 0);

        loader->Build("jacobi");

        for (int d = 4; d < 4097; d *= 2) {

            cl_int matrix_size = d;

            loader->AddArgument(&matrix_size, 4, sizeof(cl_int));

            std::cout << std::endl << (int) matrix_size << "x" << (int) matrix_size << ": ";

            cl_float *A = (cl_float *) std::malloc(matrix_size * matrix_size * sizeof(cl_float));
            cl_float *b = (cl_float *) std::malloc(matrix_size * sizeof(cl_float));
            cl_float *x1 = (cl_float *) std::malloc(matrix_size * sizeof(cl_float));
            cl_float *x2 = (cl_float *) std::malloc(matrix_size * sizeof(cl_float));

            init_diag_dom_near_identity_matrix(matrix_size, A);
            for (int i = 0; i < matrix_size; i++) { b[i] = (cl_float) ((std::rand()%51)/100.0); }

            for (int i = 0; i < matrix_size; i++) { x1[i] = (cl_float) 0; }
            for (int i = 0; i < matrix_size; i++) { x2[i] = (cl_float) 0; }

#ifdef PRINT_MTX
            printMatrix(matrix_size, matrix_size, A);
            std::cout << "*x=\n";
            printMatrix(matrix_size, 1, b);
#endif

            cl_mem buffer_A = loader->AddBuffer(CL_MEM_READ_ONLY, 0, matrix_size * matrix_size * sizeof(cl_float));
            cl_mem buffer_b = loader->AddBuffer(CL_MEM_READ_ONLY, 1, matrix_size * sizeof(cl_float));
            cl_mem buffer_x1 = loader->AddBuffer(CL_MEM_READ_ONLY, 2, matrix_size * sizeof(cl_float));
            cl_mem buffer_x2 = loader->AddBuffer(CL_MEM_READ_WRITE, 3, matrix_size * sizeof(cl_float));

            loader->WriteBuffer(buffer_A, A, 0, matrix_size * matrix_size * sizeof(cl_float));
            loader->WriteBuffer(buffer_b, b, 1, matrix_size * sizeof(cl_float));
            loader->WriteBuffer(buffer_x1, x1, 2, matrix_size * sizeof(cl_float));
            loader->WriteBuffer(buffer_x2, x2, 3, matrix_size * sizeof(cl_float));


            int iteration = 0;

            const size_t global[2] = {(size_t) matrix_size, (size_t) matrix_size};

            do {

                loader->Run(1, NULL, global);

                loader->ReadBuffer(buffer_x1, 2, matrix_size * sizeof(cl_float), x1);
                loader->ReadBuffer(buffer_x2, 3, matrix_size * sizeof(cl_float), x2);

#ifdef PRINT_MTX
                std::cout << "\n\nx=\n";
                printMatrix(matrix_size, 1, x2);
#endif

                cl_mem buffer_temp = buffer_x2;
                buffer_x2 = buffer_x1;
                buffer_x1 = buffer_temp;

                loader->ReWriteBuffer(buffer_x1, x1, 2, matrix_size * sizeof(cl_float));
                loader->ReWriteBuffer(buffer_x2, x2, 3, matrix_size * sizeof(cl_float));

                iteration++;

            } while (iteration < ITERATIONS);

            loader->PrintProfileInfo();

            free(A);
            free(b);
            free(x1);
            free(x2);

        }

        delete loader;

        return 0;

    } catch (const std::exception &e) {

        std::cout << std::flush;
        std::cerr << std::flush << "Exception thrown: " << e.what();

        return -1;
    }
}

