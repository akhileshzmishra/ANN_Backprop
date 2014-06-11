#ifndef __ANNINTERFACE___
#define __ANNINTERFACE___

#pragma once
#include "ANNHeader.h"
#include "NeuralNetwork.h"
class ANNInterface
{
	typedef ANNetwork NETWORK;
	NETWORK _network;
	bool _isTrained;
public:
	ANNInterface(int* networkconfi, int networkLevels, int userinputVectorDim);
	~ANNInterface(void);
	void TrainNetwork(vector<vector<double> >& Examples, vector<vector<double> >& TargetValues);
	void NetworkResult(vector<double>& input, vector<double>& ResultContainer);
	void ChangeMomentumConstant(double value);
	void ChangeLearningRate(double value);
	bool IsTrained();
};




#endif