#include "directedgraph.h"
#include "exception.h"

// default constructor
directedgraph::directedgraph()
{
	NumberNode = 0;
}
	
// constructor that reads directly from an input file the information;
directedgraph::directedgraph(std::string _Filename ) 
{
	std::ifstream _Dev;
    _Dev.open(_Filename.c_str(),std::ios::in);

    if (!_Dev.is_open()) throw Quynh::exception("Cannot open file", errno);
		
    std::string _Item;
	int _NumberNode;
    while(_Dev.good())
    {
        std::vector<std::string> _Token;
        getline(_Dev, _Item,'\n');
        if(_Item.empty()) continue;
        _Token = utils::tokenize(_Item,' ');
        if(_Token[1] =="node") {
            _NumberNode = utils::stringtolong(_Token[2]); // get the number of node of graph
        }
        else if (_Token[0] =="end"){ break;}
    }

	// One information is passed to memory now
	NumberNode = _NumberNode;
    
    int _NumberNode2 = 0;
    while(_Dev.good() && (_NumberNode2 < _NumberNode))
    {
        std::vector<std::string> _Token;
        std::getline(_Dev, _Item,'\n');
        if(_Item.empty()) continue;
        _Token = utils::tokenize(_Item,' ');
        if(_Token[0]=="l")
		{
            int n = _Token.size();
            std::vector<int> *_AdjacientList = new std::vector<int>;
            for(int i = 1;i < n; i++)
            {
              _AdjacientList->push_back(utils::stringtolong(_Token[i]));//got a list of adjacient node;
            }
            (this->InList).push_back(*_AdjacientList);
            delete _AdjacientList;
        }
    }
    _Dev.close();
    std::cout<<"Get data for InList and NumberNode from file successfully!" << std::endl;
}


// copy constructor
directedgraph::directedgraph( const directedgraph &_TmpCopyGraph )
{
	this->InList = _TmpCopyGraph.InList;
	this->NumberNode = _TmpCopyGraph.NumberNode;
}


// destructor
directedgraph::~directedgraph()
{

}

void directedgraph::operator=(const directedgraph & _TmpGraph)
{
	NumberNode = _TmpGraph.NumberNode;
	InList = _TmpGraph.InList;
}

// to get the InList;
std::vector<std::vector<int>> directedgraph::getInList() const
{
	return this->InList;
}
	
	
// get the number of node;
int directedgraph::getNumberNode() const
{
	return NumberNode;
}
void directedgraph::setNumberNode(int _TmpVar)
{
	NumberNode = _TmpVar;
}


void directedgraph::loadGraphFromFile(std::string _Filename ) // from adjacent list, the output is gonna be the outindegree
{
    if(!InList.empty()) (InList).clear();
	std::ifstream _Dev;
    _Dev.open(_Filename.c_str(), std::ios::in);

    if(!_Dev.is_open()) throw Quynh::exception("Cannot open file", errno);
    
    std::string _Item;
	int _NumberNode;
    while(_Dev.good())
    {
        std::vector<std::string> _Token;
        getline(_Dev, _Item,'\n');
        if(_Item.empty()) continue;
        _Token = utils::tokenize(_Item,' ');
        if(_Token[1] =="node") {
            _NumberNode = utils::stringtolong(_Token[2]); // get the number of node of graph
        }
        else if (_Token[0] =="end"){ break;}
    }
    
	// the first information is passed to the memory
	NumberNode = _NumberNode;

    int _NumberNode2 = 0;
    while(_Dev.good() && (_NumberNode2 < _NumberNode) )
    {
        std::vector<std::string> _Token;
        getline(_Dev, _Item,'\n');
        if(_Item.empty()) continue;
        _Token = utils::tokenize(_Item,' ');
        if(_Token[0]=="l"){
            int _Nn = (int)_Token.size();
            std::vector<int> *_AdjacientList = new std::vector<int>;
            for(int i = 1;i<_Nn; i++)
            {
              _AdjacientList->push_back(utils::stringtolong(_Token[i]));//got a list of adjacient node;
            }
            (this->InList).push_back(*_AdjacientList);
            delete _AdjacientList;
        }
    }
    _Dev.close();
	std::cout << "Loaded file successfully! " << std::endl;
}


// export directed graph in form of adjmat
void directedgraph::printGraph2File(std::string _FileName ) const
{
	std::ofstream _Dev;
	_Dev.open(_FileName, std::ios::out);

	_Dev << "number node " << NumberNode << std::endl;
	_Dev << "end header!" << std::endl;
	for (int i = 0; i < (int)this->InList.size(); i++)
	{
		size_t _SizeOfRowI = (this->InList)[i].size();
		for (size_t j = 0; j<_SizeOfRowI; j++)
		{
			_Dev << (this->InList)[i][j] << " ";

		}
		_Dev << std::endl;
	}
	_Dev.close();
    std::cout << "Write to file successfully!" << std::endl;
}
