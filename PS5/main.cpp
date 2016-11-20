#include <iostream>

#include "ClWrapper.h"

#include "time_ms.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"
#define VALUE cl_int
#define WORKGROUP_SIZE 256

// Fill array with: { 1, 2, 3, ..., N}
void fillArray(VALUE *vec, size_t N);

void printArray(VALUE *vec, size_t N);

// Gauß'sche Summenformel: 1+2+3+...+N == (1/2) * N * (N+1)
void validateSum(VALUE sum, size_t N);

VALUE iterativeReduction(VALUE *vector, size_t N);

int main() {

    try {
        cl_uint N = 4096;
        VALUE *vec = (VALUE *) malloc(sizeof(VALUE) * N);
        VALUE *result = (VALUE *) malloc(sizeof(VALUE) * N);
        for (size_t i = 0; i < N; i++) {
            result[i] = 0;
        }

        fillArray(vec, N);

        VALUE iterativeSum = iterativeReduction(vec, N);
        validateSum(iterativeSum, N);

        ClWrapper cl("../reduction.c", 0);
        cl.Build("reduction");

        cl::Buffer b_array = cl.AddBuffer(CL_READ_ONLY_CACHE, 0, sizeof(VALUE) * N);
        cl::Buffer b_result = cl.AddBuffer(CL_READ_WRITE_CACHE, 1, sizeof(VALUE) * N);

        cl.WriteBuffer(b_array, vec, 0, sizeof(VALUE) * N);
        cl.WriteBuffer(b_result, result, 1, sizeof(VALUE) * N);

        cl.kernel.setArg(2, sizeof(VALUE) * N, NULL);

        cl::NDRange global(N);
        cl::NDRange local(WORKGROUP_SIZE);

        for (int i = N; i > 0; i /= WORKGROUP_SIZE) {
            cl.Run(local, global);

            cl.ReadBuffer(b_result, 2, sizeof(VALUE) * N, result);
            cl.kernel.setArg(0, b_result);
        }

        validateSum(result[0], N);
        std::cout << "Time for parallel reduction: " << cl.getTotalExecutionTime() << "ms" << std::endl;


        return 0;
    } catch (const cl::Error &e) {
        std::cerr << "OpenCL exception: " << e.what() << " : " << ClWrapper::get_error_string(e.err());
    } catch (const std::exception &e) {

        std::cout << std::flush;
        std::cerr << std::flush << "Exception thrown: " << e.what();

        return -1;
    }
}

VALUE iterativeReduction(VALUE *vector, size_t N) {

    unsigned long start_time = time_ms();
    VALUE sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += vector[i];
    }

    printf("Time for iterative reduction: %9lu ms\n", time_ms() - start_time);

    return sum;
}


void printArray(VALUE *vec, size_t N) {
    size_t i = 0;
    printf("Array: { ");
    for (; i < N - 1; ++i) {
        printf("%d, ", vec[i]);
    }
    printf("%d}\n", vec[i]);
}


void fillArray(VALUE *vec, size_t N) {
    for (size_t i = 1; i <= N; ++i) {
        vec[i - 1] = (VALUE) i;
    }
}

void validateSum(int sum, size_t N) {
    VALUE check = (VALUE) (N * (N + 1));
    if (sum == check / 2) {
        printf("validation correct: %d == %d\n", check / 2, sum);
    } else {
        printf("validation fail: %d != %d\n", check / 2, sum);
    }
}

#pragma clang diagnostic pop