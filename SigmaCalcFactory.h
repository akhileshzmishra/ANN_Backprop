#ifndef _ISIGMA_CALC_FACTORY__
#define _ISIGMA_CALC_FACTORY__


#pragma once
#include "SigmaCalculation.h"




class SigmaCalcFactory
{
public:
	SigmaCalcFactory(void);
	~SigmaCalcFactory(void);
	ISigmaCalculation* CreateSigma(SigmaCalType type = CommonSigma);
};





#endif