/*
Fabio Vittorini
K-Means in Cuda
*/

#include <vector>
#include <thrust/host_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
using namespace std;

#ifndef CUDA_KMEANS_H_
#define CUDA_KMEANS_H_

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while(0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

/* Run k-Means */
float* run(float* dataset, int k, float threshold, int size, int dims, int blockDim);


#endif /* CUDA_KMEANS_H_ */
