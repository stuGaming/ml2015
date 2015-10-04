/*
                                                                           
   (               )                                        )              
 ( )\     )     ( /(       (                  (  (     ) ( /((             
 )((_) ( /(  (  )\())`  )  )(   (  `  )   (   )\))( ( /( )\())\  (   (     
((_)_  )(_)) )\((_)\ /(/( (()\  )\ /(/(   )\ ((_))\ )(_)|_))((_) )\  )\ )  
 | _ )((_)_ ((_) |(_|(_)_\ ((_)((_|(_)_\ ((_) (()(_|(_)_| |_ (_)((_)_(_/(  
 | _ \/ _` / _|| / /| '_ \) '_/ _ \ '_ \/ _` |/ _` |/ _` |  _|| / _ \ ' \)) 
 |___/\__,_\__||_\_\| .__/|_| \___/ .__/\__,_|\__, |\__,_|\__||_\___/_||_|  
                    |_|           |_|         |___/                         

 For more information on back-propagation refer to:
 Chapter 18 of Russel and Norvig (2010).
 Artificial Intelligence - A Modern Approach.
 */
#include <vector>
#include <iostream>
#include "CNeuralNet.h"
#include <math.h>   
#include <Windows.h>
using namespace std;


int hiddenSize;
int outSize;
int inSize;

double learn;
double square;

struct node {
	double in1;
	double in2;
	double nd;
	double out1;
	double out2;
	double err;
};

vector<node> hiddenNodes;
vector<double> inNodes;
vector<double> outNodes;

/**
 The constructor of the neural network. This constructor will allocate memory
 for the weights of both input->hidden and hidden->output layers, as well as the input, hidden
 and output layers.
*/
CNeuralNet::CNeuralNet(uint inputLayerSize, uint hiddenLayerSize, uint outputLayerSize, double lRate, double mse_cutoff)
	//you probably want to use an initializer list here
{
	hiddenSize = hiddenLayerSize;
	outSize = outputLayerSize;
	inSize = inputLayerSize;
	learn = lRate;
	square = mse_cutoff;
	initWeights();
	//TODO
}
/**
 The destructor of the class. All allocated memory will be released here
*/
CNeuralNet::~CNeuralNet() {
	//TODO
}
/**
 Method to initialize the both layers of weights to random numbers
*/
void CNeuralNet::initWeights(){
	float temp2;
	for (int i = 0; i < hiddenSize; i++) {
		node temp;
		temp2 = (((double)rand()) / (RAND_MAX));
		temp.in1 = ((((double)rand()) / (RAND_MAX))*2)-1;
		temp.in2 = ((((double)rand()) / (RAND_MAX)) * 2) - 1;
		temp.out1 = ((((double)rand()) / (RAND_MAX)) * 2) - 1;
		temp.out2 = ((((double)rand()) / (RAND_MAX)) * 2) - 1;
		temp.nd = 0;
		temp.err = 0;
		
		hiddenNodes.push_back(temp);
	}
	for (int i = 0; i < inSize; i++) {
		inNodes.push_back(0);
		outNodes.push_back(0);

	}

	//TODO
}
/**
 This is the forward feeding part of back propagation.
 1. This should take the input and copy the memory (use memcpy / std::copy)
 to the allocated _input array.
 2. Compute the output of at the hidden layer nodes 
 (each _hidden layer node = sigmoid (sum( _weights_h_i * _inputs)) //assume the network is completely connected
 3. Repeat step 2, but this time compute the output at the output layer
*/

double CNeuralNet::sig(double x) {
	double t = 1/(1+exp(-x));
	return t;
}


void CNeuralNet::feedForward(const double * const inputs) {
	double temp=0;
	double temp2=0;
	
	for (int i = 0; i < hiddenSize; i++) {
		
		hiddenNodes[i].nd = inputs[0] * hiddenNodes[i].in1 + inputs[1] * hiddenNodes[i].in2;
		hiddenNodes[i].nd = sig(hiddenNodes[i].nd);
	}
	
	
	for (int i = 0; i < hiddenSize; i++) {
		temp += hiddenNodes[i].nd*hiddenNodes[i].out1;
		temp2 += hiddenNodes[i].nd*hiddenNodes[i].out2;
	}
	inNodes[0] = inputs[0];
	inNodes[1] = inputs[1];
	outNodes[0] = sig(temp);
	outNodes[1] = sig(temp2);
	
	  //TODO
}
/**
 This is the actual back propagation part of the back propagation algorithm
 It should be executed after feeding forward. Given a vector of desired outputs
 we compute the error at the hidden and output layers (allocate some memory for this) and
 assign 'blame' for any error to all the nodes that fed into the current node, based on the
 weight of the connection.
 Steps:
 1. Compute the error at the output layer: sigmoid_d(output) * (difference between expected and computed outputs)
    for each output
 2. Compute the error at the hidden layer: sigmoid_d(hidden) * 
	sum(weights_o_h * difference between expected output and computed output at output layer)
	for each hidden layer node
 3. Adjust the weights from the hidden to the output layer: learning rate * error at the output layer * error at the hidden layer
    for each connection between the hidden and output layers
 4. Adjust the weights from the input to the hidden layer: learning rate * error at the hidden layer * input layer node value
    for each connection between the input and hidden layers
 5. REMEMBER TO FREE ANY ALLOCATED MEMORY WHEN YOU'RE DONE (or use std::vector ;)
*/
void CNeuralNet::propagateErrorBackward(const double * const desiredOutput){
	double error1 = outNodes[0]*(1-outNodes[0])*(desiredOutput[0]-outNodes[0]);
	double error2 = outNodes[1] * (1 - outNodes[1])*(desiredOutput[1] - outNodes[1]);

	for (int i = 0; i < hiddenSize; i++) {
		//double temp = (hiddenNodes[i].out1*error1);
		//temp += (hiddenNodes[i].out2*error2);
		hiddenNodes[i].err = hiddenNodes[i].nd*(1 - hiddenNodes[i].nd)*(error1*hiddenNodes[i].out1 + error2*hiddenNodes[i].out2);
		//hiddenNodes[i].err = temp*hiddenNodes[i].nd;
	}

	for (int i = 0; i < hiddenSize; i++) {
		hiddenNodes[i].out1 += learn*hiddenNodes[i].nd *error1;
		hiddenNodes[i].out2 += learn*hiddenNodes[i].nd *error2;
	}
	
	for (int i = 0; i < hiddenSize; i++) {
		hiddenNodes[i].in1 += learn*hiddenNodes[i].err *inNodes[0];
		hiddenNodes[i].in2 += learn*hiddenNodes[i].err *inNodes[1];
	}

	

	/*
	double error1 = sig(outNodes[0])*(1-outNodes[0])*(desiredOutput[0]-outNodes[0]) ;//calculate error delta1 and 2
	double error2 = sig(outNodes[1]) * (1 - outNodes[1])*(desiredOutput[1] - outNodes[1]);
	cout << "HELLO" << endl;
	TCHAR szbuffer[32];
	sprintf(szbuffer, "%f", error2);

	OutputDebugString(szbuffer);
	OutputDebugString("\n");
	
	for (int i = 0; i < hiddenSize; i++) {//calculate the error at the hidden nodes
		hiddenNodes[i].err = hiddenNodes[i].nd*(1 - hiddenNodes[i].nd)*(hiddenNodes[i].out1*error1 + hiddenNodes[i].out2 * error2);
	}

	for (int i = 0; i < hiddenSize; i++) {
		double t1 = hiddenNodes[i].nd*learn*error1;//delta weight changes
		double t2 = hiddenNodes[i].nd*learn*error2;
		hiddenNodes[i].out1 += t1;//Add delta changes to weights
		hiddenNodes[i].out2 += t2;
	}

	for (int i = 0; i < hiddenSize; i++) {
		hiddenNodes[i].in1 = (hiddenNodes[i].in1) + (learn*(hiddenNodes[i].err*inNodes[0]));
		hiddenNodes[i].in2 = (hiddenNodes[i].in2) + (learn*(hiddenNodes[i].err*inNodes[0]));
	}

	*/
	//TODO
}
/**
This computes the mean squared error
A very handy formula to test numeric output with. You may want to commit this one to memory
*/
double CNeuralNet::meanSquaredError(const double * const desiredOutput){
	
	double sum = 0;
	for (int i = 0; i < outSize; i++) {
		double err = desiredOutput[i] - outNodes[i];
		sum += err*err;
	}
	
	/*TODO:
	sum <- 0
	for i in 0...outputLayerSize -1 do
		err <- desiredoutput[i] - actualoutput[i]
		sum <- sum + err*err
	return sum / outputLayerSize
	*/
	TCHAR szbuffer[32];
	sprintf(szbuffer, "%f", sum/outSize);

	OutputDebugString(szbuffer);
	OutputDebugString("\n");
	return sum/outSize;
}
/**
This trains the neural network according to the back propagation algorithm.
The primary steps are:
for each training pattern:
  feed forward
  propagate backward
until the MSE becomes suitably small
*/
void CNeuralNet::train(const double** const inputs,
		const double** const outputs, uint trainingSetSize) {
	cout << "works" << endl;
	for (int i = 0; i < trainingSetSize; i++) {
		feedForward(inputs[i]);
		
		propagateErrorBackward(outputs[i]);
		meanSquaredError(outputs[i]);
	}
	//TODO
}
/**
Once our network is trained we can simply feed it some input though the feed forward
method and take the maximum value as the classification
*/
uint CNeuralNet::classify(const double * const input){
	feedForward(input);
	
	if (outNodes[0] >= outNodes[1]) {
		return outNodes[0];

	}
	else {
		return -1;
	}
	//return 1; //TODO: fix me
}
/**
Gets the output at the specified index
*/
double CNeuralNet::getOutput(uint index) const{
	return 0; //TODO: fix me
}