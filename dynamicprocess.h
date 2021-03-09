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
	virtual void setTimeSeries() = 0; // pure virtual procedure to be inherited from concrete classes such as forestfire
	std::vector<std::vector< int > > getTimeSeries() const; // to get information about time series
													  // of dynamical process
	void setMaxTimeStep(int _TmpVar);
	int	 getMaxTimeStep() const;
protected:
	std::vector<std::vector<int>> TimeSeries; // store information about dynamical process
	int MaxTimeStep;
};


#endif