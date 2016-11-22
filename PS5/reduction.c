#define VALUE int

__kernel void reduction_v1(
        __const __global VALUE *buffer,
        __global VALUE *result
) {

    int sum = 0;
    for (int i = 0; i < get_global_size(0); i++) {
        sum += buffer[i];
    }

    result[0] = sum;
}

__kernel void reduction_v2(
        __const __global VALUE *buffer,
        __global VALUE *result,
        __local VALUE *scratch
) {

    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    int global_size = get_global_size(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);

    if (global_index < global_size) {
        scratch[local_index] = buffer[global_index];
    } else {
        scratch[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = local_size / 2; offset > 0; offset >>= 1) {

        if (local_index < offset) {
            VALUE other = scratch[local_index + offset];
            VALUE mine = scratch[local_index];
            scratch[local_index] = mine + other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[group_id] = scratch[0];
    }

}

