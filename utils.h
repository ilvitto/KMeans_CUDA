/*
Fabio Vittorini
K-Means in Cuda
*/

#include <string>
#include <vector>

#ifndef UTILS_H_
#define UTILS_H_

class FileReader{

public:
	float** loadFromFile(std::string filename, int size, int dims);
};

double wtime(void);

#endif /* UTILS_H_ */
