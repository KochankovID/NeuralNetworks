#pragma once
#include "NeyronPerceptron.h"
#include "Functors.h"
#include "PLearns.h"
#include "Weights.h"

#define D_Perceptron NeyronPerceptron<double, double>
#define F_Perceptron NeyronPerceptron<float, float>
#define I_Perceptron NeyronPerceptron<int, int>