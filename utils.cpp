/*
Fabio Vittorini
K-Means in Cuda
*/

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "utils.h"

using namespace std;

float** FileReader::loadFromFile(string filename, int size, int dims) {
    float** dataset = new float*[size];
    ifstream file;
    file.open(filename.c_str());
    int index = 0;

	for(int i = 0; i < size; i++){
		dataset[i] = new float[dims];
		for(int j = 0; j < dims; j++){
			file >> dataset[i][j];
			index++;
		}

	}

    file.close();
    cout << "Length: " << index << endl;
    return dataset;
}

double wtime(void) {
    double          now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              /* in seconds */
               ((double)etstart.tv_usec) / 1000000.0;  /* in microseconds */
    return now_time;
}
