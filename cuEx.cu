#ifndef CUDA_H
#define CUDA_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>


using namespace std;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 

__global__ void mykernel( int *a, int *b )
{
	a[0] += b[0];
}

extern "C" 
void Dosomething()
{
	int a = 5; int b = 9;
	int *d_a, *d_b; // device pointer
	
	cudaMalloc(&d_a, sizeof(int));
	cudaMalloc(&d_b, sizeof(int));

	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
	
	mykernel <<< 16, 16 >>> (d_a, d_b);

	cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << " The answer is " << a << "\n";

	cudaFree(d_a);
	cudaFree(d_b);
}

__global__ void kernel( const int *a, const int *b, int *c, int Num )
{
	int idx = blockIdx.x;
	if(idx < Num)
	{
		c[idx] = a[idx] + b[idx];
	}
}
extern "C" 
void addtwoVector( const int *a,  const int *b, int *c, int Num )
{
	int *d_a, *d_b; // device pointer
	int *d_c;
	//cout << "The number of element is " << N << endl;
	cudaMalloc(/*(void **)*/&d_a, Num*sizeof(int));
	cudaMalloc(/*(void **)*/&d_b, Num*sizeof(int));
	cudaMalloc(/*(void **)*/&d_c, Num*sizeof(int));

	cudaMemcpy(d_a, a, Num*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, Num*sizeof(int), cudaMemcpyHostToDevice);
	
	kernel <<< Num, 1 >>> (d_a, d_b, d_c, Num);

	cudaMemcpy(c, d_c, Num*sizeof(int), cudaMemcpyDeviceToHost);

	//std::cout << " The answer is " << a << "\n";
	for(int i = 0; i < 10; i ++ )
	{
		cout << c[i] << " ";
	}
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

#define Num 41452143 //8223 * 71 * 71

#define N 50000000
__global__ void AddLong( const int* a, const int* b, int* c, int n)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if( idx < n)
	{
		c[idx] = a[idx] + b[idx];
		idx += blockDim.x*gridDim.x;
	}
}

extern "C"
void addTwolongVector(const int* a, const int* b, int *c, int n )
{
	int *d_a, *d_b, *d_c;
	HANDLE_ERROR( cudaMalloc(&d_a, n*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc(&d_b, n*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc(&d_c, n*sizeof(int) ) );

	HANDLE_ERROR(cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR(cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice) );

	AddLong <<<256, 256>>> (d_a, d_b, d_c, n);

	HANDLE_ERROR(cudaMemcpy( c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost) );
	//std::cout << " The answer is " << a << "\n";
	for(int i = 0; i < 10; i ++ )
	{
		cout << c[i] << " ";
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

#define DIM 1000

struct cuComplex
{
	float r;
	float i;
	__device__ cuComplex( float a, float b ) : r(a), i(b){}
	__device__ float magnitude2()
	{
		return r*r + i*i;
	}
	__device__ cuComplex operator*( const cuComplex &a )
	{
		return cuComplex( r*a.r - i*a.i, i*a.r + r*a.i );
	}
	__device__ cuComplex operator+( const cuComplex &a )
	{
		return cuComplex( r + a.r, i + a.i );
	}

};

__device__ int julia( int x, int y )
{
	const float scale = 1.5;
	float jx = scale * (float)( DIM/2 - x )/ (DIM/2);
	float jy = scale * (float)( DIM/2 - y )/ (DIM/2);
	cuComplex c( -0.8, 0.156 );
	cuComplex a( jx, jy );
	for(int i = 0; i < 200; i ++ )
	{
		a = a*a + c;
		if( a.magnitude2() > 1000 )
		{
			return 0;
		}
	}
	return 1;
}
__global__ void Modify( unsigned char *ptr )
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y*gridDim.x;
	int JuliaValue = julia(x, y);
	ptr[offset*4 + 0] = 255*JuliaValue;
	ptr[offset*4 + 1] = 255*JuliaValue;
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 0;
}



// I have to flatten the array of matrix to 1D array
// For the array of set, I will have to save one additional array indicating number of element in each set
// The output should be and 1D array then I will have to store it in a 2D array (or vector);



#define MaxNumbNoise 8000 // the maximum so far could be 80000 for whole range 0 to 1, actuall 8223

#define MaxNumbNode 100 // the maximum so far could be 1000 for human connectome

#define MaxNumbClusterings 15 // the maximum rarely goes beyond this number, depending on how users chooses the number of clusterings

// to check if a entry belongs to a cluster in the clusterIndex
// input: i is the number to check if it belongs to the cluster[index + clusterIndex*MaxNumbNode];
// *cluster is the array to store all the clusters
// clusterIndex is the index th cluster in the set of clusters
// input cluster is a global variable in GPU
// Here I assume each cluster has MaxNumbNode
// output: just check if entry i belongs to the cluster number clusterIndex

__device__ bool belong_ToSet(int i, const int *cluster, int clusterIndex)
{
	bool ReturnValue = false;
	for(int index = 0; index < MaxNumbNode; index ++)
	{
		if(i == cluster[index + clusterIndex*MaxNumbNode ])
		{
			ReturnValue = true;
			return ReturnValue;
		}
	}
	return ReturnValue;
}


// The kernel to compute the average co_activation for one noise for all the clusters
// input: g_Co_matrix is the list of coactivation matrices with respect to noise which is a global variable in GPU
// input: row, column is the number of row, the number of column for each matrix. Normally row = column.
// input: g_Clustering is the array which store the information of cluster which is a global variable in GPU
// input: NumberClustering is the number of clusters
// output: store all the co-activation matrix
__global__ void compute_coactivation_average_one_parameter( const float *g_Co_matrix, const int row, const int column, const  int *g_Clustering, float *g_output, int NumberClustering)
{
	// How to determine the location of g_Co_Matrix for this threads
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// How to determine how many matrix one thread can calculate
	while(index < MaxNumbNoise )
	{// Only the block with index smaller than MaxNumbNoise has to work
		// The matrix will be including the element from  index*row*column to (index + 1)*row*colomn
		// Note that Co_matrix[index][j][k] = g_Co_matrix[index*row*column + j*row + k]
		// the emelement i of the clustering m clustering[m][i] = g_Clustering[i + m*MaxNumbNode], we just stop until we have one element = -1;
		// and the output[m][index], which is the average of cluster m, at noise i and output[m][index] = g_output[m + index*MaxNumbClusterings] 
		for(int clusterIndex = 0; clusterIndex < NumberClustering; clusterIndex ++)
		{
			float ReturnValue = 0.0f;
			float numberOfpair = 0.0f;
			for(int i = 0; i < row ; i ++)
			{
				for(int j = i+1; j < column; j ++)
				{
					// check if i and j belong to g_Clustering at [clusterIndex]
					bool I_belong_clusterIndex = belong_ToSet(i, g_Clustering, clusterIndex);
					bool J_belong_clusterIndex = belong_ToSet(j, g_Clustering, clusterIndex);
					if(I_belong_clusterIndex && J_belong_clusterIndex)
					{
						ReturnValue += g_Co_matrix[index*row*column + i*row + j];
						numberOfpair += 1.0f;
					}
				}
			}// end computing for this matrix
			if(numberOfpair != 0.0f) ReturnValue = ReturnValue/numberOfpair;
			g_output[index*NumberClustering + clusterIndex] = ReturnValue; // that's correct!!!!!
		} // End computing for each cluster at this noise
		index += blockDim.x * gridDim.x;
	} //end for this thread	
}


// Input: the coactivation matrix list of all noise setting
// Input: clustering list which include all the clustering
// output is the ensemble of average of coactivation evolving by noises
extern "C"
void Average_Coactivation( const vector<vector<vector<float>>> &CoAct_matrixList, const vector<vector<int>> &clusteringList, vector<vector<float>> &output )
{
	int TheNumbOfNoise = (int) CoAct_matrixList.size(); // Why do I need that?
	//cout << "The number of noises is " << TheNumbOfNoise << endl;
	int SizeOfMatrix = (int)(CoAct_matrixList[0]).size();
	//cout << "The size of matrix is " << SizeOfMatrix << endl;
	int NumberOfClustering = (int)clusteringList.size();
	//cout << "The number of clustering is " << NumberOfClustering << endl;
	// should delete after copy the data over to device from host; // I hope this computer can handle that, otherwise I 
	float *ArraynizeTheTwoDMatrix = new float[TheNumbOfNoise*SizeOfMatrix*SizeOfMatrix]; 
	// need to setup again the memory thingy, maximum memory for heap in visual studio :)
	for(int i = 0; i < TheNumbOfNoise; i ++)
	{
		for(int j = 0; j < SizeOfMatrix; j ++)
		{
			for(int k = 0; k < SizeOfMatrix; k ++)
			{
				ArraynizeTheTwoDMatrix[i*SizeOfMatrix*SizeOfMatrix + j*SizeOfMatrix + k] = CoAct_matrixList[i][j][k];
			}
		}
	}// I assume that I build the array correctly

	float *dev_CoActMat;
	// Allocate memory for the dev_CoActMat in device
	HANDLE_ERROR(cudaMalloc(&dev_CoActMat, TheNumbOfNoise*SizeOfMatrix*SizeOfMatrix*sizeof(float))); 
	HANDLE_ERROR(cudaMemcpy(dev_CoActMat, ArraynizeTheTwoDMatrix, TheNumbOfNoise*SizeOfMatrix*SizeOfMatrix*sizeof(float), cudaMemcpyHostToDevice)); 
	// Deleting the array in host
	delete [] ArraynizeTheTwoDMatrix; // mission accomplished, its duty is done

	// now need to work with the clustering array in host
	// I have to allocate that amount of memory, because I have no ideas how  users are going to choose their clusterings.
	int *ArrayClusteringList = new int[NumberOfClustering*MaxNumbNode]; 
	for(int i = 0; i < NumberOfClustering*MaxNumbNode; i ++)
	{
		// clusteringList[m][n] = ArrayClusteringList[n + m*MaxNumbNode]; 
		// How about ArrayClusteringList[p] = clusteringList[x][y]; 1. What are x, y? 2. What if clusteringList[x][y] doesn't exist?
		// 1. p = y + x*MaxNumbNode; so p/MaxNumbNode = y/MaxNumbNode + x, get floor [p/MaxNumbNode] = x because [y/MaxNumbNode] = 0; y = p - x*MaxNumbNode
		// 2. Check if x < clusteringList.size(), yes then check if y < clusteringList[x].size(), if yes assgin ArrayClusteringList[p] = clusteringList[x][y], else = -1, else = -1;
		int index_cluster = (int)floor((float)i/(float)MaxNumbNode);
		int index_element_inCluster = i - index_cluster*MaxNumbNode;
		if(index_cluster < NumberOfClustering)
		{
			if(index_element_inCluster < (int)clusteringList[index_cluster].size())
			{
				ArrayClusteringList[i] = clusteringList[index_cluster][index_element_inCluster];
			}
			else ArrayClusteringList[i] = -1;
		}
		else ArrayClusteringList[i] = -1;
	}

	// Copy the clustering array from host to device
	int *dev_ArrayClusteringList;
	HANDLE_ERROR(cudaMalloc(&dev_ArrayClusteringList, NumberOfClustering*MaxNumbNode*sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(dev_ArrayClusteringList,  ArrayClusteringList, NumberOfClustering*MaxNumbNode*sizeof(int), cudaMemcpyHostToDevice ));
	// Deleting the array in host
	delete [] ArrayClusteringList;

	// Now I had already in the GPU memory the list of matrix and the list of clustering data, the only thing I have to do now is to allocate memory for the output data
	float *dev_output_average_coactivation_by_noises;
	HANDLE_ERROR(cudaMalloc(&dev_output_average_coactivation_by_noises, TheNumbOfNoise*NumberOfClustering*sizeof(float)));
	//cout <<"Allocate memory succesfully"<< endl;
	
	// use MaxNumbNoise blocks and 1 threads per block to run the kernel
	compute_coactivation_average_one_parameter<<< 128,128 >>>(dev_CoActMat, SizeOfMatrix, SizeOfMatrix, dev_ArrayClusteringList, dev_output_average_coactivation_by_noises, NumberOfClustering); 
	
	// I'm done computing till this step, now I have to copy from device to host then delete memory in GPU 
	float *output_average_coativation_by_noises = new float[TheNumbOfNoise*NumberOfClustering];
	HANDLE_ERROR(cudaMemcpy(output_average_coativation_by_noises, dev_output_average_coactivation_by_noises, TheNumbOfNoise*NumberOfClustering*sizeof(float), cudaMemcpyDeviceToHost));

	// Now modify back to my data structure output
	if(!output.empty()) output.clear();
	for(int i = 0; i < NumberOfClustering; i ++)
	{
		vector<float> Output_Average_Coactivation_clusterI;
		for(int j = 0; j < TheNumbOfNoise; j ++)
		{
			float EntryIJ = output_average_coativation_by_noises[j*NumberOfClustering + i];
			Output_Average_Coactivation_clusterI.push_back(EntryIJ);
		}
		output.push_back(Output_Average_Coactivation_clusterI);
	}
	//cout << "The number of noise after computing is " << (int)output.size() << endl;
	// Now I can delete output_average in host
	delete [] output_average_coativation_by_noises; 
	
	// Now I can delete memory in GPU
	cudaFree(dev_CoActMat);
	cudaFree(dev_ArrayClusteringList);
	cudaFree(dev_output_average_coactivation_by_noises);
}

#endif