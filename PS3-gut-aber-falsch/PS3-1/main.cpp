//
// Created by Roland Gritzer on 13.10.16.
//

#include <iostream>
#include <cmath>
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

float init_func(int x, int y) {
    return 40 * std::sin((float)(16 * (2 * x - 1) * y));
}


int main() {

    try {

        ClLoader *loader = new ClLoader("../jacobi.c", 0);

        loader->Build("jacobi");

        for (cl_uint matrix_size = 4; matrix_size < 4097; matrix_size *= 2) {

            for (int iterations = 100; iterations < 100000; iterations *= 10) {

                loader->AddArgument(&matrix_size, 4, sizeof(cl_uint));

                std::cout << std::endl << (int) matrix_size << "x" << (int) matrix_size << "/" << iterations << ": ";

                cl_float *A = (cl_float *) std::malloc(matrix_size * matrix_size * sizeof(cl_float));
                cl_float *b = (cl_float *) std::malloc(matrix_size * sizeof(cl_float));
                cl_float *x1 = (cl_float *) std::malloc(matrix_size * sizeof(cl_float));
                cl_float *x2 = (cl_float *) std::malloc(matrix_size * sizeof(cl_float));

                init_diag_dom_near_identity_matrix(matrix_size, A);
                for (cl_uint i = 0; i < matrix_size; i++) { b[i] = (cl_float) ((std::rand() % 51) / 100.0); }

                for (cl_uint i = 0; i < matrix_size; i++) { x1[i] = (cl_float) 0; }
                for (cl_uint i = 0; i < matrix_size; i++) { x2[i] = (cl_float) 0; }

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

                } while (iteration < iterations);

                loader->PrintProfileInfo();

                free(A);
                free(b);
                free(x1);
                free(x2);

            }
        }

        delete loader;

        return 0;

    } catch (const std::exception &e) {

        std::cout << std::flush;
        std::cerr << std::flush << "Exception thrown: " << e.what();

        return -1;
    }
}

