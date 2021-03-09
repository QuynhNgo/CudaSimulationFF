#include "forestfire.h"

forestfire::forestfire():dynamicprocess()
{
	RecoveryProbability  = 0.0f;
	SpontaneousFiring	 = 0.0f;
}
forestfire::~forestfire()
{

}

void forestfire::setRecoveryProbability(double _TmpVar)
{
	this->RecoveryProbability = _TmpVar;
}
	
double forestfire::getRecoveryProbability() const
{
	return this->RecoveryProbability;
}
	
void forestfire::setSpontaneousFiring(double _TmpVar)
{
	this->SpontaneousFiring = _TmpVar;
}
	
double forestfire::getSpontaneousFiring() const
{
	return this->SpontaneousFiring;
}

void forestfire::setGraphAttach(const directedgraph &_TmpGraph)
{
	GraphAttach = _TmpGraph;
}

// This void simulate the dynamical process on NetWorks with 3 stages!
// The model is quite simple!
// int start_node: is the name of start node which triggers the process
// Threes stages are S: susceptible, E: excited, refractory R
// Which are updated synchronously in discrete time steps arcording to the following rules:
// 1. S to E if there is at least one excitation in its direct neighborhood,
// if not S to E with probability f
// 2. E to R
// 3. R to S with probability p;
// to easily implementation, we transform S is -1, R is 0, and E is 1;
// f, p are controlling parameters, to which users can control, interact.
// start_stage: beginning stage of start_node is E (1), remaining nodes' stage all are R;
// For each time step t; we update stage for all node, from start_node
void forestfire::setTimeSeries()
{
	if(!( this->TimeSeries ).empty()) ( this->TimeSeries ).clear(); // to make sure accumulation doesn't happen when one change parameters to run another simulation run
	srand(static_cast<unsigned int>(time(0)));;//To create random numbers.
    int _NodeNumber = GraphAttach.getNumberNode();
    std::vector<std::vector<int>> _EdgeInfo = GraphAttach.getInList();
    std::vector<std::vector<int>> _Value; // ToDo: you can replace value by timeseries;
    for(int i = 0; i < _NodeNumber; i++)
	{
        // initialize attribute for all nodes differs from start_node;
        std::vector<int> _ValueI;
        double _Probability = (double)rand()/(double) RAND_MAX;
        if(_Probability < 0.333f)   _ValueI.push_back(1);
        else if( (0.333f <= _Probability ) && ( _Probability <= 0.666f ) ) _ValueI.push_back(0);
		else _ValueI.push_back(-1);
        _Value.push_back(_ValueI);
    }

    // Done initializing stage of all nodes
    // Now then, update each time_step just by update_attribute;
    // For each node, check adjacient_list of this node,
    //considering the last entry of attribute to decide next stage of this node;
    for(int j = 1;j < MaxTimeStep; j++){
        for(int i = 0; i< _NodeNumber; i++){
           float _CurrentStage = _Value[i][j-1];
            if(_CurrentStage == 1){
                _Value[i].push_back(0);// next_time_step's_attribue;
            }
            else if(_CurrentStage ==0){
                double _Probability = (double)rand()/(double) RAND_MAX;
                if(_Probability < RecoveryProbability){ //recovery probability is p;
                    _Value[i].push_back(-1);
                }
                else _Value[i].push_back(0);
            }
            else{

                std::vector<int> _Edge1 = _EdgeInfo[i]; // this is the node to get information
                                                  //about neighbors
                int _LenghtNeighborNode1 = (int) _Edge1.size(); // the lenght of adjacient list
                                        //of node1 (the number of adjacient nodes of node1)
                bool _Excited = false;
                for(int k = 0; k< _LenghtNeighborNode1; k++)
                    {
                        // now try to access the neighbor node
                        int _NextNodeName = _Edge1[k];
                        if((_Value[_NextNodeName][j-1]==1)){
                            _Value[i].push_back(1);
                            _Excited = true;
                            break;
                        }
                    }
                if(!_Excited){
                            double _Probability = (double)rand()/(double)RAND_MAX;
                            if(_Probability < SpontaneousFiring){ //spontaneousfiring f
                                _Value[i].push_back(1);
                            }
                            else _Value[i].push_back(-1); // Else update current stage to
                            //be the next stage;
                }
            }
        }
      }// End update attribute for step_t
   this->TimeSeries = _Value;
   std::cout << "Done forest fire simulation " << std::endl;
}


// compute and Output co-activation matrix to file
// output file is stored in the file Filename;
void forestfire::computeOutputCoActivationMatrix(std::string _Filename) const
{
	int _NumberNodeGraph =  (int)TimeSeries.size();
	int _TimeStep = this->getMaxTimeStep();
	std::vector<std::vector<double>> _DistanceMatrix; // could you float instead of double to save memory
	for(int i = 0; i < _NumberNodeGraph; i ++)
	{
		std::vector<double> _RowI;
		for(int j = 0; j < _NumberNodeGraph; j ++)
		{
			_RowI.push_back(0.0);
		}
		_DistanceMatrix.push_back(_RowI);
	}
    for(int i = 0; i < _NumberNodeGraph; i++)
    {
        std::vector<int> _PhaseNodeI = (TimeSeries)[i]; //Phase stage of Node I
        for( int j= i + 1; j < _NumberNodeGraph; j++ )
        {
            float _Sum = 0.0f;
            float _ExitedI = 0.0f; float _ExitedJ = 0.0f;
            std::vector<int> _PhaseNodeJ = (TimeSeries)[j]; //Phase stage of Node J
			// discard the first 1000 time steps;
            for(int k = 1000; k < _TimeStep; k++)
            {
                if(_PhaseNodeI[k] == 1) _ExitedI += 1.0f;
                if(_PhaseNodeJ[k] == 1) _ExitedJ += 1.0f;
                if( (_PhaseNodeI[k] == _PhaseNodeJ[k]) && (_PhaseNodeI[k] == 1))
				{
					_Sum += 1.0f;
				}
            }
            _DistanceMatrix[i][j] = _Sum/(float)(_TimeStep-1000); // on step for row i of matrix disimilarity
        }
    }

	std::cout << "Done computing distance matrix" << std::endl;

	// Now write to file, it's slow to separate the process, but it's okay now
	std::ofstream _Dev;
	_Dev.open (_Filename, std::ios::out | std::ios::binary);
	for(int i = 0; i < (int)_DistanceMatrix.size(); i ++ )
	{
		for(int j = i+1; j < (int)(_DistanceMatrix[0]).size(); j ++ )
		{
			_Dev << _DistanceMatrix[i][j] << " ";
		}
		if( i != (int)_DistanceMatrix.size() - 1) _Dev << std::endl;
	}
	_Dev.close();
}