/*
Fabio Vittorini
K-Means in Cuda
*/

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_kmeans.h"
using namespace std;

__device__ const unsigned long long infty =  0x7ff0000000000000;

#define CUDA_CHECK_RESULT(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
   if(err == cudaSuccess)
	   return;
   cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at "<< file << ":" << line << endl;
   exit(1);
}

__global__ void findNearestCentroid(float* dataset, float* centroids, int* cluster, int k, int size, int dims/*) {*/ , float* sum, int* numChanged) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size) {
		int index = 0;
		double dist = infty;
		for(int i = 0; i < k; i++) {
			double currentDist = 0;
			for(int j = 0; j < dims; j++) {
				currentDist += (dataset[j*size + id] - centroids[j*k + i]) * (dataset[j*size + id] - centroids[j*k + i]);
			}
			if(currentDist < dist) {
				dist = currentDist;
				index = i;
			}
		}
		if(cluster[id] != index)
			atomicAdd(&numChanged[0], 1);
		cluster[id] = index;
	}
	else
		return;
}

__global__ void updateCentroids(float* dataset, int* cluster, int* count, float* sums, int k, int size, int dims) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size) {
		atomicAdd(&count[cluster[id]], 1);
		for(int j = 0; j < dims; j++)
			atomicAdd(&sums[j*k + cluster[id]], dataset[size*j + id]);
	}
	else
		return;
}

float* run(float* dataset, int k, float threshold, int size, int dims, int blockDim = 1024){

	// get device information
	int dev = 0;
	cudaDeviceProp deviceProp;
	CUDA_CHECK_RESULT(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CUDA_CHECK_RESULT(cudaSetDevice(dev));

	bool loop = true;

	int iteration = 0;

	float* sum;

	float* deviceDataset;

	float* centroids = new float[k * dims];
	float* deviceCentroids;

	int* cluster = new int[size];
	int* deviceCluster;

	float* sums = new float[k * dims];
	float* deviceSums;

	int* count = new int[k];
	int* deviceCount;

	int* numChanged = new int[1];
	int* deviceNumChanged;

	for(int i = 0; i < k; i++) {
		count[i] = 0;
		int randomPos = rand()%size;
		for(int j = 0; j < dims; j++) {
			centroids[j*k + i] = dataset[randomPos*j + i];
			sums[j*k + i] = 0.0;
		}
	}

	dim3 block (blockDim);
	dim3 grid ((size + block.x - 1) / block.x);

	// Create and tranfer dataset to device
	CUDA_CHECK_RESULT(cudaMalloc((void**)&deviceDataset, size * dims * sizeof(float)));
	CUDA_CHECK_RESULT(cudaMemcpy(deviceDataset, dataset, size * dims * sizeof(float), cudaMemcpyHostToDevice));

	// Create and tranfer clusters to device
	CUDA_CHECK_RESULT(cudaMalloc((void**)&deviceCluster, size * sizeof(int)));
	CUDA_CHECK_RESULT(cudaMemcpy(deviceCluster, cluster, size * sizeof(int), cudaMemcpyHostToDevice));

	// Create centroids variable to device
	CUDA_CHECK_RESULT(cudaMalloc((void**)&deviceCentroids, k * dims * sizeof(float)));

	CUDA_CHECK_RESULT(cudaMalloc((void**)&sum, size * sizeof(float)));

	// Count
	CUDA_CHECK_RESULT(cudaMalloc((void**)&deviceCount, k * sizeof(int)));

	// Sums
	CUDA_CHECK_RESULT(cudaMalloc((void**)&deviceSums, k * dims * sizeof(float)));

	CUDA_CHECK_RESULT(cudaMalloc((void**)&deviceNumChanged, sizeof(int)));

	do {
		std::cout << "\r\t  Iteration " << iteration << flush;
		loop = true;
		numChanged[0] = 0;

		// Tranfer to devices
		CUDA_CHECK_RESULT(cudaMemcpy(deviceCentroids, centroids, k * dims * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RESULT(cudaMemcpy(deviceCount, count, k * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK_RESULT(cudaMemcpy(deviceSums, sums, k * dims * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RESULT(cudaMemcpy(deviceNumChanged, numChanged, sizeof(int), cudaMemcpyHostToDevice));

		findNearestCentroid<<< grid, block >>>(deviceDataset, deviceCentroids, deviceCluster, k, size, dims, sum, deviceNumChanged);

		cudaThreadSynchronize();
		cudaDeviceSynchronize();
		checkLastCudaError();

		updateCentroids<<< grid, block >>>(deviceDataset, deviceCluster, deviceCount, deviceSums, k, size, dims);

		cudaThreadSynchronize();
		cudaDeviceSynchronize();
		checkLastCudaError();

		//Get clusters
		CUDA_CHECK_RESULT(cudaMemcpy(cluster, deviceCluster, size * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RESULT(cudaMemcpy(count, deviceCount, k * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RESULT(cudaMemcpy(sums, deviceSums, k * dims * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RESULT(cudaMemcpy(numChanged, deviceNumChanged, sizeof(int), cudaMemcpyDeviceToHost));

		cout << " Changed: " << numChanged[0];


		for(int i = 0; i < k; i++) {
			for(int j = 0; j < dims; j++){
				pow(abs(sums[j*k + i]/count[i] - centroids[j*k + i]), 2) < threshold ? loop *= true : loop *= false;
				if(count[i] > 0)
					centroids[j*k + i] = sums[j*k + i]/count[i];
				// Reset sums
				sums[j*k + i] = 0.0;
			}
			// Reset clusters dimension
			count[i] = 0;
		}

	}while(!loop && iteration++ < 100);
	/*}while(numChanged[0] < 5 && iteration++ < 100);*/

	/* Free all variables */
	free(cluster);
	free(sums);
	free(count);
	free(numChanged);

	CUDA_CHECK_RESULT(cudaFree(deviceDataset));
	CUDA_CHECK_RESULT(cudaFree(deviceCluster));
	CUDA_CHECK_RESULT(cudaFree(deviceCentroids));
	CUDA_CHECK_RESULT(cudaFree(deviceCount));
	CUDA_CHECK_RESULT(cudaFree(deviceSums));
	CUDA_CHECK_RESULT(cudaFree(deviceNumChanged));

	return centroids;
}
