#include "Sigmoid.h"
#include "SigmaCalcFactory.h"

const double InitWeight = 0.15;

Sigmoid::Sigmoid(int UserInputs, SigmaCalType type):_weights(UserInputs, InitWeight), _sigmaCalculation(SigmaCalcFactory().CreateSigma(type)), _outputStore(0.0)
{
	RandomizeWeights();
}

void Sigmoid::RandomizeWeights()
{
	srand((unsigned)(time(NULL)));
	for(int i = 0; i < (signed int)_weights.size(); i++)
	{
		_weights[i] = (double)(rand())/(RAND_MAX/2) - 1;
	}
}

Sigmoid::Sigmoid():_sigmaCalculation(SigmaCalcFactory().CreateSigma(CommonSigma))
{
}

Sigmoid::~Sigmoid(void)
{
	if(_sigmaCalculation)
		delete _sigmaCalculation;
}
Sigmoid::Sigmoid(const Sigmoid& other)
{
	_weights          = other._weights;
	_sigmaCalculation = SigmaCalcFactory().CreateSigma(other._sigmaCalculation->Type());
}



double Sigmoid::Calculate(vector<double>& input)
{
	if(input.size() != _weights.size())
		return 0.0;
	return (_outputStore = _sigmaCalculation->Result(_weights, input));
}
void Sigmoid::operator = (const Sigmoid& other)
{
	if(this == &other)
		return;
	_weights          = other._weights;
	_sigmaCalculation = SigmaCalcFactory().CreateSigma(other._sigmaCalculation->Type());
}

void Sigmoid::AssignWeights(vector<double>& weights)
{
	_weights = weights;
}
void Sigmoid::ChangeWeights(vector<double>& weights)
{
	for(int i = 0; i < (signed int)_weights.size(); i++)
	{
		_weights[i] += weights[i];
	}
}
void Sigmoid::ChangeCalculationType(SigmaCalType type)
{
	if(_sigmaCalculation)
		delete _sigmaCalculation;

	_sigmaCalculation = SigmaCalcFactory().CreateSigma(type);
}

double Sigmoid::Output()
{
	return _outputStore;
}
int Sigmoid::InputVectorDim()
{
	return _weights.size();
}
double Sigmoid::Weight(int component)
{
	return _weights[component];
}