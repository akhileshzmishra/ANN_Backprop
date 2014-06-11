#ifndef __NEURALNETWORK___
#define __NEURALNETWORK___

#pragma once
#include "ANNHeader.h"
#include "Sigmoid.h"
typedef vector<double> InputVector;
const double LEARNING_RATE = 0.03;



class ANNetwork
{
	class ANNLayerNode 
	{
		Sigmoid _sigmoid;
		vector<double> _weightChange;
		vector<double> _prevWeight;
		double _deltaCalculated;
	public:
		ANNLayerNode(int userinput);
		~ANNLayerNode();
		ANNLayerNode(const ANNLayerNode& other);
		void operator = (ANNLayerNode& other);
		inline Sigmoid& NodeSigmoid();
		inline double& WeightChange(int i);
		inline double& PrevWeight(int i);
		inline double& Delta();
		inline int UserVectorDim();
		void ApplyWeightChanges();
	};
	class ANNLayers
	{
		vector<ANNLayerNode> _nodes;
		int _layerid;
		vector<double> _input;
		ANNLayers* _nxtLayer;
		int _inputSet;
		ANNLayers* _prevLayer;
		double _learningCnst;
		double _momentumCnst;
	public:
		ANNLayers(int userinput, int layerid, int numnodes, double learningConst, double momentumCnst);
		ANNLayers();
		~ANNLayers();
		ANNLayers(const ANNLayers& other);
		void operator = (ANNLayers& other);
		inline int LayerID();
		inline int LayerNodeNum();
		inline double Input(int i);
		void setInput(double value, int i);
		void setInput(vector<double>& input);
		void Fire();
		void CalculateErr(vector<double>& target);
		void CalculateErr();
		inline ANNLayers*& NextLayer();
		inline ANNLayers*& PrevLayer();
		inline double Delta(int i);
		inline void ChangeLearningConst(double value);
		inline void ChangeMomentumConst(double value);
		inline double NodeWeight(int nodenum, int inputnum);
	public:
		void ApplyWeight();
		void SetWeight(int nodeid, vector<double>& weight);
		inline int InputSize();
		void GetOutput(vector<double>& output);
	};
	
	typedef vector<ANNLayers> LayerArray;	
	LayerArray _layers;
	int _inputVectorSize;
	double _learningConstant;
	double _momentumConstant;
public:
	ANNetwork(int userinput, int layers, int* layerconfig, double learningCnt = LEARNING_RATE);
	~ANNetwork();
	void Train(vector<vector<double>>& userinput, vector<vector<double>>& targetValue);
	vector<double> CalculateResult(vector<double>& input);
	void SetWeight(int layerid, int nodeid, vector<double>& value);
	void ChangeLearningRate(double value);
	void ChangeMomentum(double value);
private:
	void Fire(vector<double>& userinput, vector<double>& targetValue);
	void CalculateError(InputVector& targetValue);
	void ChangeWeight();
};
#endif