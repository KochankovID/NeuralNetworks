#pragma once
#include "NeyronPerceptron.h"
#include "Functors.h"
#include "PLearns.h"
#include "Weights.h"

#define DD_Perceptron NeyronPerceptron<double, double>
#define ID_Perceptron NeyronPerceptron<int, double>
#define DI_Perceptron NeyronPerceptron<double, int>
#define II_Perceptron NeyronPerceptron<int, int>