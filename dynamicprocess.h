/* File: dynamicprocess.h
 * Author: Quynh Quang
 * Created at 1:41 pm
 * on October 13th, 2015
 * Bremen, Jacobs University
 */

#ifndef DYNAMICPROCESS_H
#define DYNAMICPROCESS_H
#include <vector>

class dynamicprocess
{
public:
	dynamicprocess();
	~dynamicprocess();
	// pure virtual procedure to be inherited from concrete classes such as forestfir
	virtual void setTimeSeries() = 0; 
	// to get information about time series
	// of dynamical process
	std::vector<std::vector< int > > getTimeSeries() const; 
	void setMaxTimeStep(int _TmpVar);
	int	 getMaxTimeStep() const;
protected:
	// store information about dynamical process
	std::vector<std::vector<int>> TimeSeries; 
	int MaxTimeStep;
};


#endif