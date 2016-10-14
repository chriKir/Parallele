__kernel void matrix(const int M, const int N, const int K,
                      const __global float *A,
                      const __global float *B,
                      __global float *C) {

  const int globalRow = get_global_id(0);
  const int globalCol = get_global_id(1);
  float acc = 0.0f;

  for (int k = 0; k < K; k++) {
    acc += A[k * M + globalRow] * B[globalCol * K + k];
  }

  C[globalCol * M + globalRow] = acc;

}