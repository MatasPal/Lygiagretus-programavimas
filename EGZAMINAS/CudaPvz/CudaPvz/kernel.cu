#define __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iomanip>
#include <fstream>
#include <iostream>

using namespace std;
__global__ void printNumbers(int* maxValue1)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 100)
    {
        printf("%d\n", tid);

        atomicMax(maxValue1, tid);
    }
}

__global__ void printSquares(int* maxValue1)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 100)
    {
        printf("%d\n", tid * tid);
    }

    atomicMax(maxValue1, tid * tid);
}

int main() {
    int* max_value;

    cudaMalloc((void**)&max_value, sizeof(int));

    int initMax = -1;
    cudaMemcpy(max_value, &initMax, sizeof(int), cudaMemcpyHostToDevice);

    printNumbers << <2, 50 >> > (max_value);
    cudaDeviceSynchronize();

    int maxVal0;
    cudaMemcpy(&maxVal0, max_value, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Max value: %d\n", maxVal0);

    printSquares << <2, 50 >> > (max_value);
    cudaDeviceSynchronize();

    int maxVal;
    cudaMemcpy(&maxVal, max_value, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Max value: %d\n", maxVal);

    cudaFree(max_value);

    return 0;
}
