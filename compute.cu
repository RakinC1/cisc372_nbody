#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

// Kernel 1: Compute pairwise accelerations
// Each thread computes accels[i*n + j] - the acceleration on object i due to object j
__global__ void computeAccelerations(vector3 *d_hPos, double *d_mass, vector3 *d_accels, int n) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n && j < n) {
		int idx = i * n + j;
		if (i == j) {
			FILL_VECTOR(d_accels[idx], 0, 0, 0);
		} else {
			vector3 distance;
			for (int k = 0; k < 3; k++) {
				distance[k] = d_hPos[i][k] - d_hPos[j][k];
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;
			FILL_VECTOR(d_accels[idx],
					accelmag * distance[0] / magnitude,
					accelmag * distance[1] / magnitude,
					accelmag * distance[2] / magnitude);
		}
	}
}

// Kernel 2: Sum columns of acceleration matrix to get total acceleration on each object
// Each thread sums all accelerations affecting object i
__global__ void sumAccelerations(vector3 *d_accel_sum, vector3 *d_accels, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n) {
		FILL_VECTOR(d_accel_sum[i], 0, 0, 0);
		for (int j = 0; j < n; j++) {
			int idx = i * n + j;
			for (int k = 0; k < 3; k++) {
				d_accel_sum[i][k] += d_accels[idx][k];
			}
		}
	}
}

// Kernel 3: Update velocities and positions based on accelerations
__global__ void updatePositionsVelocities(vector3 *d_hVel, vector3 *d_hPos, vector3 *d_accel_sum, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n) {
		for (int k = 0; k < 3; k++) {
			d_hVel[i][k] += d_accel_sum[i][k] * INTERVAL;
			d_hPos[i][k] += d_hVel[i][k] * INTERVAL;
		}
	}
}

// Main compute function called from nbody.c
extern "C" void compute() {
	int n = NUMENTITIES;
	
	// Configure grid and block dimensions
	int blockSize = 16;
	
	// For kernel 1: 2D grid for n x n acceleration matrix
	dim3 threads1(blockSize, blockSize);
	dim3 blocks1((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
	
	// For kernels 2 and 3: 1D grid for n objects
	int blocks_1d = (n + blockSize - 1) / blockSize;
	
	// Allocate temporary device memory for acceleration matrix
	vector3 *d_accels_temp;
	cudaMalloc((void**)&d_accels_temp, sizeof(vector3) * n * n);
	
	// Allocate temporary device memory for acceleration sums
	vector3 *d_accel_sum;
	cudaMalloc((void**)&d_accel_sum, sizeof(vector3) * n);
	
	// Kernel 1: Compute pairwise accelerations
	computeAccelerations<<<blocks1, threads1>>>(d_hPos, d_mass, d_accels_temp, n);
	cudaDeviceSynchronize();
	
	// Kernel 2: Sum columns of acceleration matrix
	sumAccelerations<<<blocks_1d, blockSize>>>(d_accel_sum, d_accels_temp, n);
	cudaDeviceSynchronize();
	
	// Kernel 3: Update positions and velocities
	updatePositionsVelocities<<<blocks_1d, blockSize>>>(d_hVel, d_hPos, d_accel_sum, n);
	cudaDeviceSynchronize();
	
	// Free temporary device memory
	cudaFree(d_accels_temp);
	cudaFree(d_accel_sum);
}

