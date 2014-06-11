#include "NeuralNetwork.h"

const int INPUT_LAYER = 0;
const int SECOND_LAYER = INPUT_LAYER + 1;
const double Momentum = 0.0;

ANNetwork::ANNLayerNode::ANNLayerNode(int userinput):_sigmoid(userinput),_weightChange(userinput, 0.0), _prevWeight(userinput, 0.0), _deltaCalculated(0.0)
{
}
ANNetwork::ANNLayerNode::~ANNLayerNode()
{
}
ANNetwork::ANNLayerNode::ANNLayerNode(const ANNetwork::ANNLayerNode& other)
{
	_sigmoid         = other._sigmoid;
	_weightChange    = other._weightChange;
	_prevWeight      = other._prevWeight;
	_deltaCalculated = other._deltaCalculated;
}
Sigmoid& ANNetwork::ANNLayerNode:: NodeSigmoid()
{
	return _sigmoid;
}
void ANNetwork::ANNLayerNode:: operator = (ANNLayerNode& other)
{
	if(this == &other)
		return;
	_sigmoid         = other._sigmoid;
	_weightChange    = other._weightChange;
	_prevWeight      = other._prevWeight;
	_deltaCalculated = other._deltaCalculated;
}
double& ANNetwork::ANNLayerNode::WeightChange(int i)
{
	return _weightChange[i];
}
double& ANNetwork::ANNLayerNode::PrevWeight(int i)
{
	return _prevWeight[i];
}
double& ANNetwork::ANNLayerNode::Delta()
{
	return _deltaCalculated;
}

int ANNetwork::ANNLayerNode::UserVectorDim()
{
	return _weightChange.size();
}
void ANNetwork::ANNLayerNode::ApplyWeightChanges()
{
	_sigmoid.ChangeWeights(_weightChange);
}
ANNetwork::ANNLayers::ANNLayers(int userinput, int layerid, int numnodes, double learningConst, double momentumCnst):_nodes(numnodes, ANNLayerNode(userinput)), _layerid(layerid), _input(userinput, 0.0), _nxtLayer(0), _inputSet(0), _prevLayer(0), _learningCnst(learningConst), _momentumCnst( momentumCnst)
{
}
ANNetwork::ANNLayers::ANNLayers()
{
}
ANNetwork::ANNLayers::~ANNLayers()
{
}
ANNetwork::ANNLayers::ANNLayers(const ANNLayers& other)
{
	_nodes = other._nodes;
	_layerid = other._layerid;
	_input   = other._input;
	_nxtLayer = other._nxtLayer;
	_inputSet = other._inputSet;
	_prevLayer = other._prevLayer;
	_learningCnst = other._learningCnst;
	_momentumCnst = other._momentumCnst;
}
void ANNetwork::ANNLayers::operator = (ANNLayers& other)
{
	if(this == &other)
		return;
	_nodes = other._nodes;
	_layerid = other._layerid;
	_input   = other._input;
	_nxtLayer = other._nxtLayer;
	_inputSet = other._inputSet;
	_prevLayer = other._prevLayer;
	_learningCnst = other._learningCnst;
	_momentumCnst = other._momentumCnst;
}

int ANNetwork::ANNLayers::LayerID()
{
	return _layerid;
}
int ANNetwork::ANNLayers::LayerNodeNum()
{
	return (signed int)_nodes.size();
}

double ANNetwork::ANNLayers::Input(int i)
{
	return _input[i];
}

void ANNetwork::ANNLayers::Fire()
{
	for(int i = 0; i < (signed int)_nodes.size(); i++)
	{
		double output = _nodes[i].NodeSigmoid().Calculate(_input);
		//cout<<"Layerid "<<_layerid<<" Node "<<i<<" Output: "<<output<<endl;
		if(_nxtLayer)
		{
			_nxtLayer->setInput(output, i);
		}
	}
}
void ANNetwork::ANNLayers::setInput(double value, int i)
{
	if(i >= (signed int)_input.size() || i < 0)
		return;
	_input[i] = value;
	//cout<<"Layerid "<<_layerid<<" Input place "<<i<<" Input: "<<value<<endl;
	_inputSet++;
	if(_inputSet == _input.size())
	{
		Fire();
		_inputSet = 0;
	}
}
double ANNetwork::ANNLayers::NodeWeight(int nodenum, int inputnum)
{
	if(nodenum >= (signed int)_nodes.size() || nodenum < 0)
		return 0.0;
	return _nodes[nodenum].NodeSigmoid().Weight(inputnum);
}
void ANNetwork::ANNLayers::setInput(vector<double>& input)
{
	if(_input.size() != input.size())
		return;
	//cout<<"Layerid Input "<< _layerid<<endl;
	for(int i = 0; i < (signed int)input.size(); i++)
	{
		_input[i] = input[i];
		//cout<<_input[i]<<endl;
	}
	Fire();
	_inputSet = 0;
}
ANNetwork::ANNLayers*& ANNetwork::ANNLayers::NextLayer()
{
	return _nxtLayer;
}
ANNetwork::ANNLayers*& ANNetwork::ANNLayers::PrevLayer()
{
	return _prevLayer;
}
void ANNetwork::ANNLayers::CalculateErr(vector<double>& target)
{
	//Calculate for the output layer
	//----------------------------------------------------------------------------------------*
	//diffDeltaNet     = dE/dNet for unit j.                                                  *
	//diffDeltaWeight  = dE/dW for unit j                                                     *
	//dE/dW            = (dE/dNet)*(dNet/dW)                                                  *
	//Net              = Sumof(W(i)*x(i)) where i is the ith component of vector and x is the * 
	//                   input                                                                *
	//dE/dW            = x*(dE/dNet) for unit j                                               *
	//Delta(W)         = -(dE/dW)*learningConstant for ith input on unit j                    *
	//dE/dNet          = -(t - o)*o*(1 - o) where t is target and o is output of jth unit     *
	//----------------------------------------------------------------------------------------*
	if(target.size() != _nodes.size())
		return;
	if(!_nxtLayer)
	{
		for(int i = 0; i < (signed int)target.size(); i++)
		{
			double output = _nodes[i].NodeSigmoid().Output();
			_nodes[i].Delta() = -(target[i] - output)*output*(1 - output);
			//cout<<_nodes[i].Delta()<<endl;
			for(int j = 0; j < _nodes[i].UserVectorDim(); j++)
			{
				double prevWC = _nodes[i].PrevWeight(j);
			    _nodes[i].PrevWeight(j) = -1.0*_nodes[i].Delta()*_learningCnst*_input[j];
				_nodes[i].WeightChange(j) = _nodes[i].PrevWeight(j) + _momentumCnst*prevWC;
				//cout<<"Layerid "<<_layerid<<" Node "<<i<<"Input Number "<<j<<" WeightChange: "<<_nodes[i].WeightChange(j)<<endl;
			}
		}
		if(_prevLayer)
		{
			_prevLayer->CalculateErr();
		}
	}
}

void ANNetwork::ANNLayers::CalculateErr()
{
	//For Hidden Values
	//-----------------------------------------------------------------------------------------*
	//diffDeltaNetHidden = dE/dNet = o(1 - o)(Sumof((dE/dNet)*W) for ever input to the         *
	//                     layer.                                                              *
	//Delta(W)           = -(dE/dW)*learningConstant for ith input on unit j                   *
	//-----------------------------------------------------------------------------------------*
	if(!_nxtLayer)
		return;
	
	for(int i = 0; i < (signed int)_nodes.size(); i++)
	{
		double dwnStream = 0.0;
		double output = _nodes[i].NodeSigmoid().Output();
		for(int j = 0; j < _nxtLayer->LayerNodeNum(); j++)
		{
			dwnStream += _nxtLayer->Delta(j)*_nxtLayer->NodeWeight(j, i);
		}
		//dwnStream += _nodes[i].NodeSigmoid().Weight(i); // bias
		_nodes[i].Delta() = dwnStream*(output)*(1 - output);
		//cout<<_nodes[i].Delta()<<endl;
		for(int j = 0; j < _nodes[i].UserVectorDim(); j++)
		{
			double prevWC = _nodes[i].PrevWeight(j);
			_nodes[i].PrevWeight(j) = _nodes[i].Delta()*_learningCnst*_input[j];
			_nodes[i].WeightChange(j) = _nodes[i].PrevWeight(j) + _momentumCnst*prevWC;
			//cout<<"Layerid "<<_layerid<<" Node "<<i<<"Input Number "<<j<<" WeightChange: "<<_nodes[i].WeightChange(j)<<endl;
			
		}
	}
	if(_prevLayer)
	{
		_prevLayer->CalculateErr();
	}
}

double ANNetwork::ANNLayers::Delta(int i)
{
	if(i >= (signed int)_nodes.size() || i < 0)
		return 0.0;
	return _nodes[i].Delta();
}

void ANNetwork::ANNLayers::ChangeLearningConst(double value)
{
	_learningCnst = value;
}

void ANNetwork::ANNLayers::ChangeMomentumConst(double value)
{
	_momentumCnst = value;
}

void ANNetwork::ANNLayers::ApplyWeight()
{
	for(int i = 0; i < (signed int)_nodes.size(); i++)
	{
		_nodes[i].ApplyWeightChanges();
	}
}

void ANNetwork::ANNLayers::SetWeight(int nodeid, vector<double>& weight)
{
	if(nodeid >= (signed int)_nodes.size() || nodeid < 0)
		return;
	_nodes[nodeid].NodeSigmoid().AssignWeights(weight);
}

int ANNetwork::ANNLayers::InputSize()
{
	return _input.size();
}
void ANNetwork::ANNLayers::GetOutput(vector<double>& output)
{
	if(output.size() != _nodes.size())
		return;
	for(int i = 0; i < (signed int)_nodes.size(); i++)
	{
		output[i] = _nodes[i].NodeSigmoid().Output();
	}
}

ANNetwork::ANNetwork(int userinput, int layers, int* layerconfig, double learningCnt):_layers(layers), _inputVectorSize(userinput), _learningConstant(learningCnt), _momentumConstant(Momentum)
{
	_layers[INPUT_LAYER] = ANNLayers(userinput, INPUT_LAYER, layerconfig[INPUT_LAYER], _learningConstant, _momentumConstant) ;
	_layers[INPUT_LAYER].PrevLayer() = 0;
	for(int i = SECOND_LAYER; i < layers; i++)
	{
		_layers[i] = ANNLayers(layerconfig[i - 1], i, layerconfig[i], _learningConstant, _momentumConstant) ;
		_layers[i].PrevLayer() = &_layers[i - 1];
		_layers[i - 1].NextLayer() = &_layers[i];
	}
	_layers[_layers.size() - 1].NextLayer() = 0;
}
ANNetwork::~ANNetwork()
{
}
void ANNetwork::ChangeLearningRate(double value)
{
	_learningConstant = value;
	for(int i = 0; i < (signed int)_layers.size(); i++)
	{
		_layers[i].ChangeLearningConst(value);
	}
}
void ANNetwork::ChangeMomentum(double value)
{
	_momentumConstant = value;
	for(int i = 0; i < (signed int)_layers.size(); i++)
	{
		_layers[i].ChangeMomentumConst(value);
	}
}
void ANNetwork::Train(vector<vector<double>>& userinput, vector<vector<double>>& targetValue)
{
	if(userinput.size() != targetValue.size())
		return;
	int OutputLayer = _layers.size() - 1;
	for(int i = 0; i < (signed int)userinput.size(); i++)
	{
		if(userinput[i].size() != _layers[INPUT_LAYER].InputSize())
			continue;
		if(targetValue[i].size() != _layers[OutputLayer].LayerNodeNum())
			continue;
		Fire((InputVector)(userinput[i]), (InputVector)(targetValue[i]));
	}
}
vector<double> ANNetwork::CalculateResult(vector<double>& userinput)
{
	int OutputLayer = _layers.size() - 1;
	_layers[INPUT_LAYER].setInput(userinput);
	vector<double> output(_layers[OutputLayer].LayerNodeNum());
	_layers[OutputLayer].GetOutput(output);
	return output;
}
void ANNetwork::SetWeight(int layerid, int nodeid, vector<double>& value)
{
	if(layerid >= (signed int)_layers.size() || layerid < 0)
		return;

	_layers[layerid].SetWeight(nodeid, value);
	
}
void ANNetwork::Fire(InputVector& userinput, InputVector& targetValue)
{
	_layers[INPUT_LAYER].setInput((vector<double>)userinput);	
	CalculateError(targetValue);
}

void ANNetwork::CalculateError(InputVector& targetValue)
{
	int outputLayer = _layers.size() - 1;
	_layers[outputLayer].CalculateErr((vector<double>)targetValue);
	ChangeWeight();
}

void ANNetwork::ChangeWeight()
{
	for(int i = 0; i < (signed int)_layers.size(); i++)
	{
		_layers[i].ApplyWeight();
	}
}