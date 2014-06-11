#include "SigmaCalcFactory.h"


SigmaCalcFactory::SigmaCalcFactory(void)
{
}


SigmaCalcFactory::~SigmaCalcFactory(void)
{
}

ISigmaCalculation* SigmaCalcFactory::CreateSigma(SigmaCalType type)
{
	ISigmaCalculation* calc = 0;
	switch(type)
	{
	case CommonSigma:
		calc = new CommSigCalc();
		break;
	default:
		break;
	}
	return calc;
}