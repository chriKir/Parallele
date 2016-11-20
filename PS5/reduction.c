#define VALUE int

/**
 *
 */
__kernel void reduction(
        __const __global VALUE *buffer,
        __global VALUE *result,
        __local VALUE *scratch
) {

    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    int global_size = get_global_size(0);
    int local_size = get_local_size(0);
//    printf("%d\n", buffer[55]);
//    printf("%d\n", local_size);

    if (global_index < global_size) {
        scratch[local_index] = buffer[global_index];
    } else {
        scratch[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

//    printf("%d:%d\n", global_index, local_index);
    for (int offset = 1; offset < local_size; offset <<= 1) {
        int mask = (offset << 1) - 1;
        if ((local_index & mask) == 0) {
            VALUE other = scratch[local_index + offset];
            VALUE mine = scratch[local_index];
            scratch[local_index] = mine + other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }

}

