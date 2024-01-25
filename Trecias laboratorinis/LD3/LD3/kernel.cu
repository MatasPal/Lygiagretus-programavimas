#define __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iomanip>
#include <fstream>
#include <iostream>


using namespace std;

const int MAX_NAME = 17;
const int RESULT_SIZE = 20;

class Car {

public:
	char manufacture[MAX_NAME];
	int year;
	double engine;

};


class InOut {
public:
	Car* Read(string& fileName, int& carCount) {
		Car* carArray = nullptr;

		ifstream file(fileName);

		if (!file.is_open()) {
			cerr << "Can not open the file - " << fileName << endl;
			return carArray;
		}

		carCount = 0;
		while (file.ignore(numeric_limits<streamsize>::max(), '\n')) {
			carCount++;
		}
		file.clear();
		file.seekg(0);

		carArray = new Car[carCount];

		for (int i = 0; i < carCount; i++) {
			char manufacture[MAX_NAME];
			int year;
			double engine;

			file >> manufacture >> year >> engine;

			strncpy(carArray[i].manufacture, manufacture, MAX_NAME);
			carArray[i].year = year;
			carArray[i].engine = engine;
		}

		file.close();

		return carArray;
	}

	void PrintCarArrayToTxt(string& fileName, Car* initialData, int initialCount, char* resultArray, int lastIndex) {
		ofstream outFile(fileName);

		outFile << "Pradiniai duomenys:\n";
		// Spausdiname pradinius duomenis
		for (int i = 0; i < initialCount; i++) {
			outFile << initialData[i].manufacture << " " << initialData[i].year << " " << initialData[i].engine << "\n";
		}

		// Atskiriamas tarpu pradinius ir rezultatų duomenis
		outFile << "\n";
		outFile << "Gauti rezultatai:\n";
		//Spausdiname rezultatus
		for (int i = 0; i < lastIndex; i++) {
			char a = resultArray[i];
			if (a != '\0') {
				outFile << a;
			}
			if ((i + 1) % RESULT_SIZE == 0 && i != lastIndex - 1) {
				outFile << "\n";
			}
		}

		if (lastIndex == 0) {
			outFile << "Result Array is empty\n";
		}
		outFile.close();
	}

};


__device__ char engineClass(double engine) {
	if (engine <= 2) {
		return 'S';
	}
	else if (engine <= 4) {
		return 'M';
	}
	else {
		return 'L';
	}
}

__device__ char yearClass(int year) {
	if (year >= 2013) {
		return 'N';
	}
	else {
		return 'O';
	}
}

__device__ char toUpperDevice(char c) {
	if (c >= 'a' && c <= 'z') {
		return c - 'a' + 'A';
	}
	return c;
}

__device__ char* calculateResult(Car car, char* result) {
	result[0] = engineClass(car.engine);
	result[1] = yearClass(car.year);
	result[2] = '-';
	for (int i = 0; i < MAX_NAME - 1; i++) {
		char a = toUpperDevice(car.manufacture[i]);
		if (a != ' ') {
			result[i + 3] = a;
		}
		else {
			break;
		}
	}
	return result;
}


__global__ void Operation(Car* deviceCars, char* resultArray, int* index, int carCount) {
	int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
	char temp_result[RESULT_SIZE];
	if (thread_index < carCount) {
		calculateResult(deviceCars[thread_index], temp_result);

		if (temp_result[0] == 'L' && temp_result[1] == 'N') {
			int write_index = atomicAdd(index, RESULT_SIZE);

			for (int i = 0; i < RESULT_SIZE; i++) {
				if (temp_result[i] != ' ') {
					resultArray[write_index + i] = temp_result[i];
				}
				else {
					break;
				}
			}
		}
	}
}



int main()
{

	int deviceCount;
	cudaGetDeviceCount(&deviceCount); // Get the number of CUDA-capable GPUs

	for (int device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		std::cout << "Device " << device << ": " << deviceProp.name << "\n";
		std::cout << "Maximum threads per block: " << deviceProp.maxThreadsPerBlock << "\n";
		std::cout << "Maximum block dimensions (x, y, z): (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")\n";
		std::cout << "Maximum grid dimensions (x, y, z): (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")\n";
	}


	setlocale(LC_CTYPE, ".UTF-8");
	string data1 = "IFF-1-8_PalujanskasM_L3_dat_1.txt"; // all data follows criteria
	string data2 = "IFF-1-8_PalujanskasM_L3_dat_2.txt"; // half data follows criteria
	string data3 = "IFF-1-8_PalujanskasM_L3_dat_3.txt"; // none data follows criteria
	string rez = "IFF-1-8_PalujanskasM_L3_rez.txt";

	if (ifstream(rez)) {
		remove(rez.c_str());
	}
	//Read to CPU
	InOut io;
	int carCount = 0;
	Car* hostCars = io.Read(data2, carCount);

	//Initial data
	io.PrintCarArrayToTxt(rez, hostCars, carCount, nullptr, 0);

	//Memory for device cars
	Car* deviceCars;
	cudaMalloc(&deviceCars, carCount * sizeof(Car));
	cudaMemcpy(deviceCars, hostCars, carCount * sizeof(Car), cudaMemcpyHostToDevice);

	//Memory for result array and index; single result inicialization

	char* deviceResults;
	int* resultIndex;

	cudaMalloc(&deviceResults, carCount * RESULT_SIZE * sizeof(char));
	cudaMalloc(&resultIndex, sizeof(int));
	cudaMemset(deviceResults, ' ', carCount * RESULT_SIZE * sizeof(char));
	cudaMemset(resultIndex, 0, sizeof(int));


	//Operation
	int blockThreads = 32;
	int blocks = (carCount + blockThreads - 1) / blockThreads;

	std::cout << "Paleistu bloku kiekis: " << blocks << "\n";
	std::cout << "Giju kiekis bloke: " << blockThreads << "\n";
	std::cout << "Is viso giju: " << blockThreads * blocks << "\n";

	Operation << < blocks, blockThreads >> > (deviceCars, deviceResults, resultIndex, carCount);


	//Copy back
	int lastIndex;
	cudaMemcpy(&lastIndex, resultIndex, sizeof(int), cudaMemcpyDeviceToHost);

	char* hostResult = new char[lastIndex];
	cudaMemcpy(hostResult, deviceResults, lastIndex * sizeof(char), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// free cuda
	cudaFree(deviceCars);
	cudaFree(deviceResults);
	cudaFree(resultIndex);

	// print
	io.PrintCarArrayToTxt(rez, hostCars, carCount, hostResult, lastIndex);


	delete[] hostResult;
	delete[] hostCars;

	return 0;
}

