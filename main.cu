/*
Fabio Vittorini
K-Means in Cuda
*/

#include <string>
#include <iostream>
#include "utils.h"
#include "cuda_kmeans.h"
using namespace std;

#define BLOCK_DIM 1024

int main(void) {

	/* Dataset info */
	const string filename = "/home/fabiovittorini/cuda-workspace/CUDAkmeans/datasets/GaussianDataset3000000.txt";
	int size = 3000000;	/*Total number of points in the dataset*/
	int dims = 2; /*Bidimensional points*/

	/* Algorithm parameters */
	float threshold = 0.00001;
	int ktest = 5;
	int ntest = 4;
	int ktests[5] = {2,3,8,16,32};
	int ntests[4] = {3000,30000,300000,3000000};
	int k;
	int n;

	/* Creation and initialization of timer */
	double iStart, iElaps;

	cout << "Reading from file..." << endl;
	FileReader r;
	float** dataset = r.loadFromFile(filename, size, dims);

	float* flattenDataset = new float[dims * size];
	for(int i = 0; i < size; i++)
		for(int j = 0; j < dims; j++)
			flattenDataset[j*size+i] = dataset[i][j];

	// Destroy temporary dataset
	free(dataset);

	std::cout << "k-Means" << std::endl;
	for (int ni = 0; ni < ntest; ni++) {
		n = ntests[ni];
		std::cout << "---------" << std::endl;
		std::cout << "n = " << n << std::endl;
		for (int ki=0; ki<ktest; ki++){
			k = ktests[ki];

			std::cout << "k = " << k << std::endl;

			srand(time(NULL));

			cout << "Starting.." << endl;

			iStart = wtime();
			float* centroids = run(flattenDataset, k, threshold, n, dims, BLOCK_DIM);
			iElaps = wtime() - iStart;

			std::cout << endl << endl;
			cout << "k-Means end" << endl;

			cout << "Elapsed time: " << iElaps << " seconds " << endl;

			std::cout << endl << endl;
			std::cout << "Centroids Coordinates" << endl << endl;

			for (int i = 0; i < k; i++) {
				for (int j = 0; j < dims; j++)
					std::cout << centroids[j*k + i] << " ";
				std::cout << std::endl;
			}
			free(centroids);
			// reset device
			cudaDeviceReset();
		}
	}

	// Destroy variables
	free(flattenDataset);

	return 0;
}
