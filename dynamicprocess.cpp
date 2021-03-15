#include "dynamicprocess.h"

dynamicprocess::dynamicprocess()
{
	MaxTimeStep = 100000;
}

dynamicprocess::~dynamicprocess()
{
}

std::vector<std::vector<int>> dynamicprocess::getTimeSeries() const
{
	return this->TimeSeries;
}

void dynamicprocess::setMaxTimeStep(int _TmpVar)
{
	this->MaxTimeStep = _TmpVar;
}
	
int	 dynamicprocess::getMaxTimeStep() const
{
	return this->MaxTimeStep;
}