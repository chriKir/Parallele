#pragma OPENCL EXTENSION cl_khr_fp64 : enable


__kernel void search(
        __global float *data,
        __global float *val,
        __global float *found,
        __global double *epsilon
) {
    int i = get_global_id(0);

    if(fabs((data[i] - val[0])) < epsilon[0]) {
        found[0] = 1;
    }
}