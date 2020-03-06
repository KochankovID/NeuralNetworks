#ifndef ARTIFICIALNN_NEYRONS_H
#define ARTIFICIALNN_NEYRONS_H

#include "Neyron.h"
#include "Functors.h"
#include "PLearns.h"
#include "Weights.h"


namespace ANN {
#define D_Perceptron Neyron<double>
#define F_Perceptron Neyron<float>
#define I_Perceptron Neyron<int>
}

#endif //ARTIFICIALNN_NEYRONS_H
