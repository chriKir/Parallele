#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void matrix(const __global float *A,
                     const __global float *B,
                     __global float *C,
                     const int l, const int m, const int n
) {

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    float acc = 0.0f;

    for (int k = 0; k < m; k++) {
        acc += A[globalRow * m + k] * B[k * n + globalCol];
    }

    C[globalRow * n + globalCol] = acc;
}
