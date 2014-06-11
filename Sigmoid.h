#ifndef _SIGMOID___
#define _SIGMOID___


#pragma once

#include "ANNHeader.h"
#include "SigmaCalculation.h"
#include "SigmaCalcFactory.h"

class Sigmoid
{	
	vector<double> _weights;
	ISigmaCalculation* _sigmaCalculation;
	double _outputStore;
public:
	Sigmoid(int UserInputs,SigmaCalType type = CommonSigma);
	Sigmoid();
	~Sigmoid(void);
	Sigmoid(const Sigmoid& other);
	void operator = (const Sigmoid& other);

public:
	int InputVectorDim();
	double Calculate(vector<double>& input);
	double Output();
	double Weight(int component);
	void ChangeWeights(vector<double>& weights);
	void AssignWeights(vector<double>& weights);
	void ChangeCalculationType(SigmaCalType type);
	void RandomizeWeights();
};




#endif