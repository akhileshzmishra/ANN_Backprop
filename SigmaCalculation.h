#ifndef _ISIGMA_CALC___
#define _ISIGMA_CALC___

#pragma once
#include "ANNHeader.h"



class ISigmaCalculation
{
public:
	ISigmaCalculation(void);
	~ISigmaCalculation(void);
	virtual double Result(vector<double>& weights, vector<double>& input) = 0;
	virtual SigmaCalType Type() = 0;
};


class CommSigCalc: public ISigmaCalculation
{
public:
	CommSigCalc();
	~CommSigCalc();
	double Result(vector<double>& weights, vector<double>& input);
	SigmaCalType Type();
private:
	double DotProduct(vector<double>& weights, vector<double>& input);
};



#endif