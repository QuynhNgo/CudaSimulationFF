/*
 * Author: Quynh Ngo
 * @Copyright: Quynh Ngo, except the two functions HandleError, define HANDLE_ERROR, and the random function in Cuda
 * @Cuda version to run forest fire model on a network
 */

#ifndef CUDA_H
#define CUDA_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "directedgraph.h"

/* These function are not mine, they belong to some books, some sites I read */
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 


/* this GPU kernel function is used to initialize the random states */
// I modified from blockIdx.x to be the index of the thread
__global__ void init(unsigned int seed, curandState_t* states) 
{
	int Index =  blockIdx.x * blockDim.x + threadIdx.x;
	/* we have to initialize the state */
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
				 Index, //blockIdx.x, /* the sequence number should be different for each core (unless you want all
								//cores to get the same sequence of numbers for some reason - use thread id! */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&states[Index]); // blockIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, unsigned int* numbers) 
{
	/* curand works like rand - except that it takes a state as a parameter */
	numbers[blockIdx.x] = curand(&states[blockIdx.x]) % 100;
}

/* CUDA's random number library uses curandState_t to keep track of the seed value
we will store a random state for every thread  */
// curandState_t* states;

/* allocate space on the GPU for the random states */
// cudaMalloc((void**) &states, N * sizeof(curandState_t));

/* invoke the GPU to initialize all of the random states */
// init<<<N, 1>>>(time(0), states);

/* allocate an array of unsigned ints on the CPU and GPU */
// unsigned int cpu_nums[N];
// unsigned int* gpu_nums;
// cudaMalloc((void**) &gpu_nums, N * sizeof(unsigned int));

/* invoke the kernel to get some random numbers */
// randoms<<<N, 1>>>(states, gpu_nums);

/* copy the random numbers back */
// cudaMemcpy(cpu_nums, gpu_nums, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
/*

/*******************************************************************************************/


/**** From now on, it's all mine ***********************************************************/

#define NumbCoactivationMatrices 6400

// think of kernel that output an array with the work of different thread
// with that the time series will be a big one that goes beyond the scope of a heap, too big 6553600 Gb.

// SO I HAVE TO DIVIDE THE GRID INTO SUB-GRIDS with the global function in CPU and have a loop to call it;
// It's doable;
// Each grid point is a thread
// that mean this thread should output the coactivation array portion


// Cuda version
// I need to collect list of co-activation matrices in form of an array
// Each thread will be in charge of a chunk
// The graph will be passed to global memory in GPU
// Each thread will handle one parameter setting for f, p
// The co-activation sub-array will be updated on the fly along side with


// This is the cuda version of forest fire model 
// param[in] graph which includes NumbNode, LinkListSize, LinkList from global memory
// param[in] f, p are the parameter values
// param[in] NumberOfTimeStep is from user-defined
// param[out] CoactivationMatrix

/*
 * @des return the index in the co-activation array which is actually only the upper right part of the matrix
 * @param[in] KernelIndex is the index of the kernel that are excecuting the function
 * @param[in] RowIndex is the row index of the element
 * @param[in] ColumnIndex is the column index of the element
 * @param[in] SizeOfMatrix is the size of the co-activation matrix
 * @note sizeMat*(sizeMat-1)/2 is the element of the upper part of the matrix;
 */	
__device__ int returnIndex(int KernelIndex, int RowIndex, int ColumnIndex, int SizeOfMatrix)
			{
				if(RowIndex == 0)
				{
					return  KernelIndex * SizeOfMatrix * (SizeOfMatrix-1)/2 + ColumnIndex - 1;
				}
				else
				{
					int Sum = 0;
					for(int i = 0; i < RowIndex - 1; i ++)
					{
						Sum += SizeOfMatrix - i - 1;
					}
					return KernelIndex * SizeOfMatrix * (SizeOfMatrix-1)/2 + Sum + ColumnIndex - RowIndex - 1;
				}
			}
	
/* Now the code for the kernel */

__global__ void calCoactivationMat(int *MaxTimeStep, 
									const int *NumbNode, const int *LinkListSize, 
									const int *LinkList,  const int *Dev_FirstLocationNeighbor, 
									int *Dev_PrevNodeState, int *Dev_CurrNodeState, curandState_t *states, float *CoactivationMatrix)
{

int Index =  blockIdx.x * blockDim.x + threadIdx.x;

while(Index < NumbCoactivationMatrices)
{
	double SpontaneousFiring = (double)threadIdx.x/(double)(double)blockDim.x;
	double RecoveryProbability = (double)blockIdx.x/(double)gridDim.x; 
	for(int j = 1;j < *MaxTimeStep; j++)
	{
		for(int i = 0; i< *NumbNode; i++)
		{
			float _CurrentStage = Dev_PrevNodeState[i + Index *(*NumbNode)]; 
			if(_CurrentStage == 1)
			{
				// next_time_step's_attribue;
				Dev_CurrNodeState[i + Index *(*NumbNode)] = 0;
			}
			else if(_CurrentStage == 0)
			{
				double Probability =  (double)(curand(&states[Index]) % 100)/100.0;
				if(Probability < RecoveryProbability)
				{ //recovery probability is p;
					Dev_CurrNodeState[i + Index *(*NumbNode)] = -1; 
				}
				else Dev_CurrNodeState[i + Index*(*NumbNode)] = 0; 
			}
			else
			{
				int LenghtNeighborNode1 = (int) LinkListSize[i];  // the lenght of adjacient list
										//of node1 (the number of adjacient nodes of node1)
				bool Excited = false;
				for(int k = 0; k< LenghtNeighborNode1; k++)
				{
					// now try to access the neighbor node
					int NextNodeName = LinkList[Dev_FirstLocationNeighbor[i] + k];
					if(Dev_PrevNodeState[NextNodeName] == 1)    
					{	
						Dev_CurrNodeState[i + Index*(*NumbNode)] = 1; 
						Excited = true;
						break;
					}
				}
				if(!Excited)
				{
					double Probability = (double)(curand(&states[Index]) % 100)/100.0; 
					if(Probability < SpontaneousFiring)
					{ 
						Dev_CurrNodeState[i + Index*(*NumbNode)] = 1;       
					}
					else Dev_CurrNodeState[i + Index*(*NumbNode)] = -1;     // Else update current stage to
					//be the next stage;
				}
			}// end else
		} // End second for loop

		// Update co-activation sum
		if(j > 1000)
		{ // discard the first 1000 time steps;
			for(int i = 0; i < *NumbNode; i++)
			{

				for( int k= i + 1; k < *NumbNode; k++ )
				{
					if( ( Dev_CurrNodeState[k + Index*(*NumbNode)] == Dev_CurrNodeState[i + Index*(*NumbNode)]) && (Dev_CurrNodeState[i + Index*(*NumbNode)] == 1))
					{
						CoactivationMatrix[returnIndex(Index, i, k, *NumbNode)] += 1;
					}	
				}
			}
		} // end if j > 100 

		for(int i = 0; i < *NumbNode; i ++)
		{
			Dev_PrevNodeState[i + Index*(*NumbNode)] = Dev_CurrNodeState[i + Index*(*NumbNode)];
		}
		
	} // End first for loop for max time step

	// Finish calculating co-activation
	for (int i = 0; i < *NumbNode; i ++)
	{
		for(int k = i + 1; k < *NumbNode; k ++)
		{
			CoactivationMatrix[returnIndex(Index, i, k, *NumbNode)] = CoactivationMatrix[returnIndex(Index, i, k, *NumbNode)]/(float)( *MaxTimeStep-1000); 
		}
	}
	//Go to increment 
  Index += blockDim.x * gridDim.x; // One thread can calculate many co-activation;	
 }// End while
}// End kernel

// Each thread handle their coactivation portion
	
/* The program */
	
extern "C" void extractCoactivationMatrices(const directedgraph &GraphAttach)
{
	
	/* allocate memory for graph in GPU */

	// pass number of node to device
	int *Host_NodeNumb = new int;
	*Host_NodeNumb = GraphAttach.getNumberNode();
	int *Dev_NodeNumb;
	HANDLE_ERROR(cudaMalloc(&Dev_NodeNumb, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(Dev_NodeNumb, Host_NodeNumb, sizeof(int), cudaMemcpyHostToDevice));

	// pass number of max time step to device
	int *Host_MaxTimeStep = new int;
	*Host_MaxTimeStep = 100000;
	int *Dev_MaxTimeStep;
	HANDLE_ERROR(cudaMalloc(&Dev_MaxTimeStep, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(Dev_MaxTimeStep, Host_MaxTimeStep, sizeof(int), cudaMemcpyHostToDevice));

	// pass array including number of neighbor nodes for each node to device
	int *Host_LinkListSize = new int[*Host_NodeNumb];
	std::vector<std::vector<int>> LinkList = GraphAttach.getInList();
	for(int i = 0; i < *Host_NodeNumb; i ++)
	{
		Host_LinkListSize[i] = (int)(LinkList[i]).size();
	}
	int *Dev_LinkListSize;
	HANDLE_ERROR(cudaMalloc(&Dev_LinkListSize, (*Host_NodeNumb)*sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(Dev_LinkListSize, Host_LinkListSize, (*Host_NodeNumb)*sizeof(int), cudaMemcpyHostToDevice));

		
	// pass linklist to device
	int ListSize = 0;
	for(int i = 0; i < *Host_NodeNumb; i ++)
	{
		ListSize += Host_LinkListSize[i];
	}

	int Index = 0;
	int *Host_LinkList = new int[ListSize];
	for(int i = 0; i < *Host_NodeNumb; i ++)
	{ 
		for(int j = 0; j < (int)LinkList[i].size(); j++)
		{
			Host_LinkList[Index] = LinkList[i][j];
			Index += 1;
		}
	}
	int *Dev_LinkList;
	HANDLE_ERROR(cudaMalloc(&Dev_LinkList, ListSize*sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(Dev_LinkList, Host_LinkList, ListSize*sizeof(int), cudaMemcpyHostToDevice));


	int *Host_FirstLocationNeighbor = new int[*Host_NodeNumb];
	for(int i = 0; i < *Host_NodeNumb; i ++)
	{
		int FirstLocationOfNodeI = 0;
		for(int j = 0; j < i; j ++)
		{
			FirstLocationOfNodeI += Host_LinkListSize[j];
		}
		Host_FirstLocationNeighbor[i] = FirstLocationOfNodeI;	
	}
	int *Dev_FirstLocationNeighbor;
	HANDLE_ERROR(cudaMalloc(&Dev_FirstLocationNeighbor, (*Host_NodeNumb)*sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(Dev_FirstLocationNeighbor, Host_FirstLocationNeighbor, (*Host_NodeNumb)*sizeof(int), cudaMemcpyHostToDevice));

	
	curandState_t* states;
	/* allocate space on the GPU for the random states */
	cudaMalloc((void**) &states, NumbCoactivationMatrices * sizeof(curandState_t));

	/* invoke the GPU to initialize all of the random states */
	init<<<128, 128>>>(time(0), states);

	// Allocate the previous step time series, and the current step time series from the host
	// for memory in the device 
	// Done initializing stage of all nodes
	// Now then, update each time_step just by update_attribute;
	// For each node, check adjacient_list of this node,
	//considering the last entry of attribute to decide next stage of this node;
	// Each thread will be in charge of one co-activation or more :)
    

	
	srand(static_cast<unsigned int>(time(0))); //To create random numbers.
	int *Host_PrevNodeState = new int[(*Host_NodeNumb)*NumbCoactivationMatrices];

	for(int k = 0; k < NumbCoactivationMatrices; k ++)
	{
		for(int i = 0; i < *Host_NodeNumb; i++)
		{
			// initialize attribute for all nodes differs from start_node;
			double _Probability = (double)rand()/(double) RAND_MAX;
			if(_Probability < 0.333f)  Host_PrevNodeState[i + k*(*Host_NodeNumb)] = 1;
			else if( (0.333f <= _Probability ) && ( _Probability <= 0.666f ) ) Host_PrevNodeState[i + k*(*Host_NodeNumb)] = 0;
			else Host_PrevNodeState[i + k*(*Host_NodeNumb)] = -1;
		}
	}
	
	int NumberOfCoactivationEntries = NumbCoactivationMatrices*(*Host_NodeNumb)*((*Host_NodeNumb)-1)/2;
	float *Dev_CoactivationArray;
	HANDLE_ERROR(cudaMalloc(&Dev_LinkListSize, (*Host_NodeNumb)*sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&Dev_CoactivationArray, NumberOfCoactivationEntries*sizeof(float)));
	int *Dev_PrevNodeState;
	HANDLE_ERROR(cudaMalloc(&Dev_PrevNodeState, (*Host_NodeNumb)*NumbCoactivationMatrices*sizeof(int)));
	int *Dev_CurrNodeState;
	HANDLE_ERROR(cudaMalloc(&Dev_CurrNodeState, (*Host_NodeNumb)*NumbCoactivationMatrices*sizeof(int)));
	// copy host data of previous state to device data
	HANDLE_ERROR(cudaMemcpy(Dev_PrevNodeState, Host_PrevNodeState, (*Host_NodeNumb)*NumbCoactivationMatrices*sizeof(int), cudaMemcpyHostToDevice));
	
	delete Host_NodeNumb;
	delete [] Host_LinkListSize;
	delete [] Host_LinkList;
	delete [] Host_FirstLocationNeighbor;

	/* done passing graph and allocate memory to device */
	
	/* Now run kernels */
	calCoactivationMat<<<128, 128>>>(Dev_MaxTimeStep, 
									Dev_NodeNumb, Dev_LinkListSize, 
									Dev_LinkList, Dev_FirstLocationNeighbor, 
									Dev_PrevNodeState, Dev_CurrNodeState, states, Dev_CoactivationArray);
	/* Done running kernel */
	
	/* Copy back to the host, write co-activation list to files */





	/* Done writing to files */
	// Now I can delete memory in GPU
	cudaFree(Dev_MaxTimeStep);
	cudaFree(Dev_NodeNumb);
	cudaFree(Dev_LinkListSize);
	cudaFree(Dev_PrevNodeState);
	cudaFree(Dev_CurrNodeState);
	cudaFree(Dev_CoactivationArray);

	std::cout << "Done forest fire simulation " << std::endl;
}

#endif