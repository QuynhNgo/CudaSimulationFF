#include<iostream>
#include<omp.h>
#include <time.h>
#include "utils.h"

#include "directedgraph.h"
//#include "forestfire.h"

// Cuda program
//extern "C" void extractCoactivationMatrices(const directedgraph &GraphAttach);

int returnIndexH(int KernelIndex, int RowIndex, int ColumnIndex, int SizeOfMatrix)
{
	if (RowIndex == 0)
	{
		return  KernelIndex * SizeOfMatrix * (SizeOfMatrix - 1) / 2 + ColumnIndex - 1;
	}
	else
	{
		int Sum = 0;
		for (int i = 0; i < RowIndex; i++)
		{
			Sum += SizeOfMatrix - i - 1;
		}
		return KernelIndex * SizeOfMatrix * (SizeOfMatrix - 1) / 2 + Sum + ColumnIndex - RowIndex - 1;
	}
}


/*
Stopwatch^ time1;
time1 = Stopwatch::StartNew();
//addtwoVector(a, b, c, N);
addTwolongVector(a, b, c, N);	
delete []a; delete [] b; delete [] c;
cout << "Elapsed time CPU: " << time2->ElapsedMilliseconds << endl;
cout << "The number of second CPU" << time2->ElapsedMilliseconds/1000.0 << endl;
delete []a; delete []b; delete []c; delete []d;
time1->Stop();
cout << "Elapsed time GPU: " << time1->ElapsedMilliseconds << endl;
cout << "The number of second GPU" << time1->ElapsedMilliseconds / 1000.0 << endl;
*/

/*
* @des CPU version of forest fire model that output the coactivation matrices file
* @param[in] StartingParameterF, ..P are the starting point of each parameters f, p. For instance f = 0.001; p= 0.1;
* @param[in] GridResolutionX is the resolution of the sampling in one direction. For instance 80, then the resolution will be 80 x 80
* @param[in] Incre is the incremental length of the sampling grid
* @param[in] GraphAttach is the input graph
*/


void cPUCoactivationMatrices(double StartingParameterF, double StartingParameterP, int GridResolutionX, double Incre, directedgraph GraphAttach)
{
	int NodeNumb = GraphAttach.getNumberNode();
	//std::cout << "The number of node is " << NodeNumb << std::endl;

	// pass array including number of neighbor nodes for each node to device
	int *LinkListSize = new int[NodeNumb];
	std::vector<std::vector<int>> LinkList = GraphAttach.getInList();
	for (int i = 0; i < NodeNumb; i++)
	{
		LinkListSize[i] = (int)(LinkList[i]).size();
	}
	
	// pass linklist to device
	int ListSize = 0;
	for (int i = 0; i < NodeNumb; i++)
	{
		ListSize += LinkListSize[i];
	}

	int _Index = 0;
	int *Host_LinkList = new int[ListSize];
	for (int i = 0; i < NodeNumb; i++)
	{
		for (int j = 0; j < (int)LinkList[i].size(); j++)
		{
			Host_LinkList[_Index] = LinkList[i][j];
			//std::cout << Host_LinkList[Index] << " ";
			_Index += 1;
		}
		//std::cout << std::endl;
	}

	int *Host_FirstLocationNeighbor = new int[NodeNumb];
	for (int i = 0; i < NodeNumb; i++)
	{
		int FirstLocationOfNodeI = 0;
		for (int j = 0; j < i; j++)
		{
			FirstLocationOfNodeI += LinkListSize[j];
		}
		Host_FirstLocationNeighbor[i] = FirstLocationOfNodeI;
		//std::cout << "First Neighbor " << i << " = " << Host_FirstLocationNeighbor[i] << " "; 	
	}
	//std::cout << std::endl;
	int NumbCoactivationMatrices = GridResolutionX * GridResolutionX;
	srand(static_cast<unsigned int>(time(0))); //To create random numbers.
	
	int *Host_PrevNodeState = new int[(NodeNumb)*NumbCoactivationMatrices];
	for (int k = 0; k < NumbCoactivationMatrices; k++)
	{
		for (int i = 0; i < NodeNumb; i++)
		{
			// initialize attribute for all nodes differs from start_node;
			double _Probability = (double)rand() / (double)RAND_MAX;
			if (_Probability < 0.333f)  Host_PrevNodeState[i + k*(NodeNumb)] = 1;
			else if ((0.333f <= _Probability) && (_Probability <= 0.666f)) Host_PrevNodeState[i + k*(NodeNumb)] = 0;
			else Host_PrevNodeState[i + k*(NodeNumb)] = -1;
		}
	}
	int *Dev_CurrNodeState = new int[(NodeNumb)*NumbCoactivationMatrices];
	//std::cout << "Number of coactivation matrices = " << (NodeNumb)*NumbCoactivationMatrices << std::endl;
	int NumberOfCoactivationEntries = NumbCoactivationMatrices*(NodeNumb)*((NodeNumb) - 1) / 2;
	float *CoactivationArray = new float[NumberOfCoactivationEntries];
	for (int m = 0; m < NumberOfCoactivationEntries; m++)
	{
		CoactivationArray[m] = 0;
	}
	std::cout << "I have allocated successfully the memory for coActivation Matrix in RAM." << std::endl;
	int MaxTimeStep = 100000;

	//std::cout << "Number of entries for the coactivationmatrices array is " << NumberOfCoactivationEntries << std::endl;
    #pragma omp parallel for
	for (int Index = 0; Index < NumbCoactivationMatrices; Index++)
	{
		int RowIndex = Index / (GridResolutionX);
		//std::cout << "RowIndex = " << RowIndex << std::endl;
		int ColumnIndex = Index % ((GridResolutionX)); // Index = RowIndex * GridResolution + ColumnIndex;
		//std::cout << "ColumnIndex = " << ColumnIndex << std::endl;
		double SpontaneousFiring = (double)RowIndex * (Incre)+ StartingParameterF;
		double RecoveryProbability = (double)ColumnIndex* (Incre)+ StartingParameterP;
		//std::cout << "f = " << SpontaneousFiring << " p = " << RecoveryProbability << std::endl;
		for (int j = 1; j < MaxTimeStep; j++)
		{
			for (int i = 0; i< NodeNumb; i++)
			{
				float _CurrentStage = Host_PrevNodeState[i + Index *(NodeNumb)];
				if (_CurrentStage == 1)
				{
					// next_time_step's_attribue;
					Dev_CurrNodeState[i + Index *(NodeNumb)] = 0;
				}
				else if (_CurrentStage == 0)
				{
					double Probability = (double)rand() / (double)RAND_MAX;
					if (Probability < RecoveryProbability)
					{ //recovery probability is p;
						Dev_CurrNodeState[i + Index *(NodeNumb)] = -1;
					}
					else Dev_CurrNodeState[i + Index*(NodeNumb)] = 0;
				}
				else
				{
					int LenghtNeighborNode1 = (int)LinkListSize[i];  // the lenght of adjacient list
																	 //of node1 (the number of adjacient nodes of node1)
					bool Excited = false;
					for (int k = 0; k< LenghtNeighborNode1; k++)
					{
						// now try to access the neighbor node
						int NextNodeName = Host_LinkList[(Host_FirstLocationNeighbor[i] + k)];
						if (Host_PrevNodeState[NextNodeName] == 1)
						{
							Dev_CurrNodeState[i + Index*(NodeNumb)] = 1;
							Excited = true;
							break;
						}
					}
					if (!Excited)
					{
						double Probability = (double)rand() / (double)RAND_MAX;
						if (Probability < SpontaneousFiring)
						{
							Dev_CurrNodeState[i + Index*(NodeNumb)] = 1;
						}
						else Dev_CurrNodeState[i + Index*(NodeNumb)] = -1;     // Else update current stage to
																			   //be the next stage;
					}
				}// end else
			} // End second for loop

			  // Update co-activation sum
			if (j > 1000)
			{ // discard the first 1000 time steps;
				for (int i = 0; i < NodeNumb; i++)
				{

					for (int k = i + 1; k < NodeNumb; k++)
					{
						if ((Dev_CurrNodeState[k + Index*(NodeNumb)] == Dev_CurrNodeState[i + Index*(NodeNumb)]) && (Dev_CurrNodeState[i + Index*(NodeNumb)] == 1))
						{
							CoactivationArray[returnIndexH(Index, i, k, NodeNumb)] += 1;
						}
					}
				}
			} // end if j > 100 

			for (int i = 0; i < NodeNumb; i++)
			{
				Host_PrevNodeState[i + Index*(NodeNumb)] = Dev_CurrNodeState[i + Index*(NodeNumb)];
			}

		} // End first for loop for max time step
		 
		// Finish calculating co-activation
		for (int i = 0; i < NodeNumb; i++)
		{
			for (int k = i + 1; k < NodeNumb; k++)
			{
				CoactivationArray[returnIndexH(Index, i, k, NodeNumb)] = CoactivationArray[returnIndexH(Index, i, k, NodeNumb)] / (float)(MaxTimeStep - 1000); 
			}
		}
		std::cout << "Done row Index " << RowIndex << ", column Index " << ColumnIndex << std::endl;
	}// End the loop for Index
	

	delete[] LinkListSize;
	delete[] Host_LinkList;
	delete[] Host_FirstLocationNeighbor;
	delete[] Host_PrevNodeState;
	delete[] Dev_CurrNodeState;

	// Print each matrix in one row of a file
	utils::printToFile(CoactivationArray, NumberOfCoactivationEntries / NumbCoactivationMatrices, NumberOfCoactivationEntries, "MatricesEnsemble.txt");

	delete[] CoactivationArray;
	
	
}


int main()
{

	directedgraph GraphInstance;
	// please put the graph file into the same folder with the execute application
	GraphInstance.loadGraphFromFile("undirected_hub_hierarchical_network.txt");
	// GPU version failed
	//extractCoactivationMatrices(GraphInstance);
	//Test(5, 3, 0.01);
	//Test(1, 2, 0.01, GraphInstance);
	// CPU version
	cPUCoactivationMatrices(0.001, 0.001, 80, 0.01, GraphInstance);
	int Stop;
	std::cin >> Stop;

	return 0;
}


/*// import the graph from file
directedgraph GraphInstance;
// please put the graph file into the same folder with the execute application
GraphInstance.loadGraphFromFile("undirected_hub_hierarchical_network.txt");

// an instance of forest fire
forestfire FFInstance;
FFInstance.setMaxTimeStep(100000);
FFInstance.setGraphAttach(GraphInstance);

// choose resolution for f, p 0.0001
// run simulation in [0.0001, 0.8] x [0.0001, 0.8],
// i.e., 8000x8000 grid
double p;

int PIndex, FIndex;
//#pragma omp parallel for private(FIndex)
for (PIndex = 0; PIndex < 7999; PIndex += 1)
{
p = 0.0001 + ((float)PIndex)*0.0001;
double f;
for (FIndex = 0; FIndex < 7999; FIndex += 1)
{
f += 0.0001 +((float)FIndex)*0.0001;
FFInstance.setRecoveryProbability(p);
FFInstance.setSpontaneousFiring(f);
FFInstance.setTimeSeries();
// prepare file name for coactivation matrix, PIndex first
std::ostringstream FileName2Write2;
FileName2Write2 << "DistMat_" << PIndex << "_" << FIndex;
// compute coactivation matrix and write it to file
FFInstance.computeOutputCoActivationMatrix(FileName2Write2.str());
std::cout << "Done matrix (" << PIndex << ", " << FIndex << ")." << std::endl;
}
}
*/
