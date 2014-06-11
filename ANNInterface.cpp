#include "ANNInterface.h"


ANNInterface::ANNInterface(int* networkconfi, int networkLevels, int userinputVectorDim):_network(userinputVectorDim, networkLevels, networkconfi), _isTrained(false)
{
}


ANNInterface::~ANNInterface(void)
{
}

void ANNInterface::TrainNetwork(vector<vector<double> >& Examples, vector<vector<double> >& TargetValues)
{
	_network.Train(Examples, TargetValues);
	_isTrained = true;
}
void ANNInterface::NetworkResult(vector<double>& input, vector<double>& ResultContainer)
{
	ResultContainer = _network.CalculateResult(input);
}
bool ANNInterface::IsTrained()
{
	return _isTrained;
}

void ANNInterface::ChangeMomentumConstant(double value)
{
	_network.ChangeMomentum(value);
}
void ANNInterface::ChangeLearningRate(double value)
{
	_network.ChangeLearningRate(value);
}