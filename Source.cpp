#include<iostream>
//#include<omp.h>

#include "directedgraph.h"
#include "forestfire.h"

// Cuda program
extern "C" void extractCoactivationMatrices(const directedgraph &GraphAttach);

int main()
{
	directedgraph GraphInstance;
	// please put the graph file into the same folder with the execute application
	GraphInstance.loadGraphFromFile("undirected_hub_hierarchical_network.txt");
	extractCoactivationMatrices(GraphInstance);

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
