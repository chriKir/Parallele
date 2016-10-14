//
// Created by roland on 13.10.16.
//

#include <iostream>
#include "ClLoader.h"

#define THREAD_BLOCK_SIZE 32

int main() {
  size_t l = 2;
  size_t m = 3;
  size_t n = 4;

  cl_float A[l][m] = {{0, 1, 2}, {3, 4, 5}};
  cl_float B[m][n] = {{0, 1, 2, 0}, {1, 2, 0, 1}, {2, 0, 6, 9}};
  cl_float C[l][n];

  for (int i = 0; i < l; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i][j] = 0;
    }
  }

  ClLoader *loader = new ClLoader("../matrix.c");

  loader->Build();

  loader->AddParameter(&l, sizeof(cl_int));
  loader->AddParameter(&m, sizeof(cl_int));
  loader->AddParameter(&n, sizeof(cl_int));

  loader->AddBuffer(CL_MEM_READ_ONLY, sizeof(A) * sizeof(cl_float));
  loader->AddBuffer(CL_MEM_READ_ONLY, sizeof(B) * sizeof(cl_float));
  cl_mem buffer_c = loader->AddBuffer(CL_MEM_WRITE_ONLY, sizeof(C) * sizeof(cl_float));

  const size_t global[2] = {THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE};
  const size_t local[2] = {l, n};

  loader->Run(local, global);

  loader->GetResult(buffer_c, sizeof(C) * sizeof(cl_float), &C);

  // print the result
  for (int i = 0; i < l; ++i) {
    std::string line = "";

    for (int j = 0; j < n; ++j) {
      line.append("[");
      line.append(std::to_string(C[i][j]));
      line.append("]\t");
    }
    std::cout << line << "\n";
  }
}
