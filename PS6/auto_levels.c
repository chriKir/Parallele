//# pragma OPENCL EXTENSION cl_amd_printf :enable

#define DTYPE_COLOR_VALUE unsigned char
#define DTYPE_SUM_VALUE unsigned long

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
        __local DTYPE_COLOR_VALUE *lsum,
        const unsigned int size
) {

    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    int global_size = get_global_size(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);

    for (size_t component = 0; component < COMPONENTS; component++) {
        size_t li = local_index + sizeof(DTYPE_COLOR_VALUE) * component;
        size_t gi = global_index + sizeof(DTYPE_COLOR_VALUE) * component;
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

//    if (global_index == 0) printf("%d\n", lmin[local_index]);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = local_size / 2; offset > 0; offset >>= 1) {

        for (size_t component = 0; component < COMPONENTS; component++) {

            if (local_index < offset && local_index < size) {

                int li = local_index + sizeof(DTYPE_COLOR_VALUE) * component;
                int gi = global_index + sizeof(DTYPE_COLOR_VALUE) * component;

                DTYPE_COLOR_VALUE mine = lmin[li];
                DTYPE_COLOR_VALUE other = lmin[li + offset];

                lmin[mine] = min(mine, other);
                lmax[mine] = max(mine, other);
                lsum[mine] = mine + other;
            }
        }


        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        for (size_t component = 0; component < COMPONENTS; component++) {

            wg_max[group_id + component] = max(wg_max[group_id + component], lmax[component]);
            wg_min[group_id + component] = min(wg_min[group_id + component], lmin[component]);
            wg_sum[group_id + component] += lsum[component];
        }
    }
}
