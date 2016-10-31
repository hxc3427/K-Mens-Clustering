#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>

__constant__ Vector2 centroids[3];

// GPU Kernel to change the centroid
__global__ void KMeansKernel( Datapoint* data_d, long n, int k )
{
	// Retrieve our coordinates in the block
	long blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
if(threadId < n){
		for(int i=0; i<k; i++)
		{
			//change of centroids == false
			data_d[threadId].altered = false;

			//checks distance of i point from all 3 centroids.
			if(centroids[i].distSq(data_d[threadId].p) < centroids[data_d[threadId].cluster].distSq(data_d[threadId].p)){
				//check the point is in right centroid area
				if(data_d[threadId].cluster != i){
					data_d[threadId].cluster = i;
					//change of centroids == true
					data_d[threadId].altered = true;
				}
			}
		}
	}
}


bool KMeansGPU( Datapoint* data, long n, Vector2* clusters, int k ){

	// Error return value
	cudaError_t status;

	// Number of bytes
	long bytes =  n * sizeof(Datapoint);

	// Pointers to the device
	Datapoint *data_d;

	// Allocate memory on the device
	status = cudaMalloc((void**) &data_d, bytes);

	// Specify the size of the grid and the size of the block
	int tile= 512;
	int max_blocks = 65535;

	//threads in a block
	dim3 dimBlock(tile, 1);

	int gridx = 1;
	int gridy = 1;
	
	if(n/tile <= max_blocks)
		gridx = ceil((float)n/tile);
	else{
		gridx = max_blocks;
		gridy = ceil((float)n/(tile*max_blocks));
	}
	dim3 dimGrid(gridx, gridy);
	bool KMeans = true;

	// Copy the host input data to the device
		status = cudaMemcpy(data_d, data, bytes, cudaMemcpyHostToDevice);

	while(KMeans){

		KMeans=false;

		
		// Copy the host centroids data to the constant memory
		status = cudaMemcpyToSymbol(centroids, clusters, k*sizeof(Vector2), 0, cudaMemcpyHostToDevice);
		
		// Launch the kernel
		KMeansKernel<<<dimGrid, dimBlock>>>(data_d, n, k);


		// Wait for completion
		cudaThreadSynchronize();		

	    // Retrieve the result
		cudaMemcpy(data, data_d, bytes, cudaMemcpyDeviceToHost);

		// Check for errors
         status = cudaGetLastError();
		if (status != cudaSuccess)
		{
			std::cout << "Kernel failed (data Memcpy) cudaMemcpyDeviceToHost: " << cudaGetErrorString(status) << 
							std::endl;
			cudaFree(data_d);
			return false;
		}

// squential code to calculate new centroid
		//index for 3 clusters
	for (int j=0; j<k; j++){
		//count no. of elements under a centroid region
			long no_of_points=0;
			//index for no of points in space
			for(long i=0; i<n; i++)
			{
				if(data[i].cluster == j){
					//add x and y of each point in a particular cluster
					clusters[j].x += data[i].p.x;
					clusters[j].y += data[i].p.y;
					no_of_points++;
				}
				//if altered==true, set kmeans=true
				if(data[i].altered==true){
					KMeans=true;
				}
			}
			//new centroid
				clusters[j].x /= no_of_points;
				clusters[j].y /= no_of_points;
		}
	}

	cudaFree(data_d);
	return true;
}