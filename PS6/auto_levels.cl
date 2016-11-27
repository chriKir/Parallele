//# pragma OPENCL EXTENSION cl_amd_printf :enable

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
        int li = local_index + sizeof(DTYPE_COLOR_VALUE) * component;
        int gi = global_index + sizeof(DTYPE_COLOR_VALUE) * component;
        if (input != 0) {
            lmin[li] = input[gi];
            lmax[li] = input[gi];
            lsum[li] = input[gi];
        } else {
            lmin[li] = wg_min[gi];
            lmax[li] = wg_max[gi];
            lsum[li] = wg_sum[gi];
        }
    }

    if (global_index == 0) printf("%d\n", lmin[local_index]);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = local_size / 2; offset > 0; offset >>= 1) {

        for (size_t component = 0; component < COMPONENTS; component++) {
            int li = local_index + sizeof(DTYPE_COLOR_VALUE) * component;
            int gi = global_index + sizeof(DTYPE_COLOR_VALUE) * component;

            if (li < offset) {
                int mine = li;
                int other = li + offset;

                lmin[li] = min(lmin[mine], lmin[other]);
                lmax[li] = max(lmax[mine], lmax[other]);
                lsum[li] = lsum[mine] + lsum[other];
            }
        }


        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        for (size_t component = 0; component < COMPONENTS; component++) {
            int li = local_index + sizeof(DTYPE_COLOR_VALUE) * component;

            wg_max[group_id + li] = max(wg_max[group_id + li], lmax[li]);
            wg_min[group_id + li] = min(wg_min[group_id + li], lmin[li]);
            wg_sum[group_id + li] += lsum[li];
        }
    }
}
