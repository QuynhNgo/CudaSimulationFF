/* File: forestfire.h
 * Author: Quynh Ngo
 * Created on October 13th, 2015
 * at 1:37 pm
 */

#ifndef FORESTFIRE_H
#define FORESTFIRE_H

#include <time.h> // to deal with time(0) in srand() in creating random number
#include <iostream>
#include <fstream>

#include "dynamicprocess.h"
#include "directedgraph.h"
#include "utils.h"


class forestfire : public dynamicprocess
{
public:
	forestfire();
	~forestfire();
	// for forestfire model only
	virtual void setTimeSeries(); 
	void setRecoveryProbability(double _TmpVar);
	double getRecoveryProbability() const;
	void setSpontaneousFiring(double _TmpVar);
	double getSpontaneousFiring() const;
	void setGraphAttach(const directedgraph &_TmpGraph);
	
	// compute and Output co-activation matrix to file
	void computeOutputCoActivationMatrix(std::string _Filename) const;

	
private:

	double RecoveryProbability;                         
	double SpontaneousFiring;
	directedgraph GraphAttach;
};

#endif