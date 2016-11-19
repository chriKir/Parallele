//
// Created by Roland Gritzer on 13.10.16.
//

#include <iostream>
#include <malloc.h>

#include "ClLoader.h"

//#define PRINT_MTX


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

int main() {

    try {

        ClLoader *loader = new ClLoader("../matrix.c", -1);

        loader->Build("matrix");

        cl_int l, m, n;


        for (int d = 4; d < 4097; d *= 2) {

            cl_int matrix_size = d;

            l = matrix_size;
            m = matrix_size;
            n = matrix_size;

            loader->setKernelArg(&l, 3, sizeof(cl_int));
            loader->setKernelArg(&m, 4, sizeof(cl_int));
            loader->setKernelArg(&n, 5, sizeof(cl_int));

            std::cout << std::endl << (int) matrix_size << "x" << (int) matrix_size << ": ";

            cl_float *A = (cl_float *) std::malloc(l * m * sizeof(cl_float));
            cl_float *B = (cl_float *) std::malloc(m * n * sizeof(cl_float));
            cl_float *C = (cl_float *) std::malloc(l * n * sizeof(cl_float));


            for (int i = 0; i < l * m; i++) { A[i] = (cl_float) (3.6 * i + i * i + 3.1); }
            for (int i = 0; i < m * n; i++) { B[i] = (cl_float) (1.2 * i + 0.01 * i * i + 13.9); }
            for (int i = 0; i < l * n; i++) { C[i] = 0.0; }

            for (int i = 0; i < l; ++i) {
                for (int j = 0; j < n; ++j) {
                    C[i * n + j] = 0;
                }
            }

#ifdef PRINT_MTX
            printMatrix(l, m, A);
            std::cout << "\n*\n\n";
            printMatrix(m, n, B);
#endif

            cl::Buffer buffer_a = loader->AddBuffer(CL_MEM_READ_ONLY, 0, l * m * sizeof(cl_float));
            cl::Buffer buffer_b = loader->AddBuffer(CL_MEM_READ_ONLY, 1, m * n * sizeof(cl_float));
            cl::Buffer buffer_c = loader->AddBuffer(CL_MEM_READ_WRITE, 2, l * n * sizeof(cl_float));

            loader->WriteBuffer(buffer_a, A, 0, l * m * sizeof(cl_float));
            loader->WriteBuffer(buffer_b, B, 1, m * n * sizeof(cl_float));
            loader->WriteBuffer(buffer_c, C, 2, l * n * sizeof(cl_float));

            cl::NDRange global((size_t) l, (size_t) n);

            loader->Run(cl::NullRange, global);

            loader->ReadBuffer(buffer_c, 2, l * n * sizeof(cl_float), C);

#ifdef PRINT_MTX
            std::cout << "\n=\n\n";
            printMatrix(l, n, C);
#endif

            loader->PrintProfileInfo();

            free(A);
            free(B);
            free(C);

        }

        delete loader;

        return 0;

    } catch (const std::exception &e) {

        std::cout << std::flush;
        std::cerr << std::flush << "Exception thrown: " << e.what();

        return -1;
    }
}

