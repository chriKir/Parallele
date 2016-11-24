# pragma OPENCL EXTENSION cl_amd_printf :enable

#define DTYPE_COLOR_VALUE uchar
#define DTYPE_SUM_VALUE ulong

#define COMPONENTS (3)

/**
 * gets the min max average value of an image by reduction
 * run number_of_pixels / WORKGROUP_SIZE times
 * @param input
 * @param result contains
 * @param scratch
 * @return
 */
__kernel void mmav_reduction(
        __const __global DTYPE_COLOR_VALUE *input,
        __global DTYPE_COLOR_VALUE *wg_min,
        __global DTYPE_COLOR_VALUE *wg_max,
        __global DTYPE_SUM_VALUE *wg_sum,
        __local DTYPE_COLOR_VALUE *lmin,
        __local DTYPE_COLOR_VALUE *lmax,
        __local DTYPE_COLOR_VALUE *lsum
) {

    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    int global_size = get_global_size(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);

    for (size_t component = 0; component < COMPONENTS; component++) {
        size_t local = local_index + sizeof(DTYPE_COLOR_VALUE) * component;
        size_t global = global_index + sizeof(DTYPE_COLOR_VALUE) * component;
        if (input != 0) {
            lmin[local] = input[global];
            lmax[local] = input[global];
            lsum[local] = input[global];
        } else {
            lmin[local] = wg_min[global];
            lmax[local] = wg_max[global];
            lsum[local] = wg_sum[global];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = local_size / 2; offset > 0; offset >>= 1) {


        if (local_index < offset) {
            int mine = local_index;
            int other = local_index + offset;


            lmin[local_index] = min(lmin[mine], lmin[other]);
            lmax[local_index] = max(lmax[mine], lmax[other]);
            lsum[local_index] = lsum[mine] + lsum[other];
            printf("%lu\n", lsum[group_id]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        wg_max[group_id] = max(wg_max[group_id], lmax[0]);
        wg_min[group_id] = min(wg_min[group_id], lmin[0]);
        wg_sum[group_id] += lsum[0];
    }
}
