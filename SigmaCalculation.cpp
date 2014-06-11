#include "SigmaCalculation.h"


ISigmaCalculation::ISigmaCalculation(void)
{
}


ISigmaCalculation::~ISigmaCalculation(void)
{
}


CommSigCalc::CommSigCalc(void)
{
}


CommSigCalc::~CommSigCalc(void)
{
}

double CommSigCalc::Result(vector<double>& weights, vector<double>& input)
{
	double dotPro = DotProduct(weights, input);
	double exponential = exp(-dotPro);
	return (double)(1.0/(1.0 + exponential));
}

double CommSigCalc::DotProduct(vector<double>& weights, vector<double>& input)
{
	double retval = 0.0;
	for(int i = 0; i < weights.size(); i++)
		retval += abs(weights[i]* input[i]);// + weights[i];
	return (retval);
}

SigmaCalType CommSigCalc::Type()
{
	return CommonSigma;
}