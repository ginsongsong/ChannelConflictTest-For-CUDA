#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "cuda.h"




#define NUM   (256*1024*1024)

#define THREADS_PER_BLOCK_X  384
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

#define PROTECT_BITS  (0xFFFF0000)

__global__ void
test_kernel( int* __restrict__ buf, int protectBits, int shrinkBits)
{

	int x = blockDim.x * blockIdx.x + threadIdx.x;

	int address;
	address = (x & protectBits) | (x & shrinkBits);

	buf[address] = x;
	//printf("address[%d] tid:%d \n ",address,x);
}


using namespace std;

int main() {

	int* hostA;

	int* deviceA;


	cudaEvent_t start, stop;

	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	float eventMs = 1.0f;


	hostA = (int*)malloc(NUM * sizeof(int));


	cudaMalloc((void**)& deviceA, NUM * sizeof(int));
	cudaMemcpy(deviceA, hostA, NUM * sizeof(int), cudaMemcpyHostToDevice);

	
	test_kernel<<<dim3(1,1,1),dim3(1,1,1),0,0>>>( deviceA , 0x0, 0x0);

	for (int i = 16; i < 64 * 1024; i = i << 1) {

		cudaEventRecord(start, 0);
		test_kernel<<<dim3(NUM/THREADS_PER_BLOCK_X, 1, 1),dim3(THREADS_PER_BLOCK_X, 1, 1),0,0>>>(deviceA,PROTECT_BITS,i - 1);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&eventMs, start, stop);

		printf("elapsed time:%f\n", eventMs);
		int bandwidth = (double)NUM * sizeof(int) / 1024 / 1024 / 1024 / (eventMs / 1000);
		printf("Shrink Size in Bytes[%ld], bandwidth %d (GB/S)\n", i*sizeof(int), bandwidth);

	}

	cudaFree(deviceA);

	free(hostA);

	return 0;
}

