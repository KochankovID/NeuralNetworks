#ifndef ARTIFICIALNN_CONVOLUTIONNETWORK_H
#define ARTIFICIALNN_CONVOLUTIONNETWORK_H

#include "NeyronCnn.h"
#include "CNNLearning.h"
#include "NeyronPerceptron.h"
#include "PerceptronLearning.h"
#include <string>
#include <vector>

struct ConvolutionLayer;
template <typename T>
class Props;


template <typename T, Props<T> props>
class ConvolutionNetwork{
public:
    // Конструктор по умолчанию
    ConvolutionNetwork();

    void startLearning(size_t epoch = props.epoch);
    void work(Matrix<T>);

    // Деструктор
    ~ConvolutionNetwork();
private:
    NeyronPerceptron<T,T> perceptron;
    NeyronCnn<T> cnn;
    PerceptronLearning<T,T> perceptronLearning;
    CNNLearning<T> cnnLearning;
    std::vector<std::vector<Filter<float> > > filters;
    std::vector<std::vector<Weights<T> > > weights;

};

template<typename T, Props<T> props>
ConvolutionNetwork<T, props>::ConvolutionNetwork() {
    perceptron = NeyronPerceptron<T,T>();
    cnn = NeyronCnn<T>();
    perceptronLearning = PerceptronLearning<T,T>(props.teacherPerceptronSpeed);
    cnnLearning = CNNLearning<T>(props.teacherCnnSpeed);

    srand(time(0));

    filters = std::vector<std::vector<Filter<float> > >(props.numberOfConvolutionLayers);
    for(int ii = 0; ii < props.numberOfConvolutionLayers; ii++){
        filters[ii] = std::vector<Filter<float> >(props.convLayers[ii].kollOfFilters);
        for (int i = 0; i < props.convLayers[ii].kollOfFilters; i++) {
            filters[ii][i] = Filter<float>(props.convLayers[ii].filterWidth, props.convLayers[ii].filterHeight);
            for (int j = 0; j < filters[ii][i].getN(); j++) {
                for (int p = 0; p < filters[ii][i].getM(); p++) {
                    if(props.isWeightsPositive){
                        filters[ii][i][j][p] = ((T) rand() / (RAND_MAX * props.coefficientInitialWeights));
                    }else {
                        filters[ii][i][j][p] = (p % 2 ? ((T) rand() / (RAND_MAX * props.coefficientInitialWeights)) : -(
                                (T) rand() / (RAND_MAX * props.coefficientInitialWeights)));
                    }
                }
            }
        }
    }




}

template <typename T>
class Props{
    static double teacherPerceptronSpeed;
    static double teacherCnnSpeed;

    static Func<T,T> func_activation;
    static Func<T,T> func_derivative;

    static size_t inputMatrixWidth;
    static size_t inputMatrixHeight;

    static size_t numberOfConvolutionLayers;
    static ConvolutionLayer convLayers[numberOfConvolutionLayers];

    static size_t numberOfHiddenDirectDistributionLayers;
    static size_t numberOfNeyronsInHiddenCurrentDistributionLayer[numberOfHiddenDirectDistributionLayers];

    static double coefficientInitialWeights;

    static size_t numberEpoch;
    static size_t numberClasses;

    static bool isWeightsPositive;
};

struct ConvolutionLayer{
    static bool maxPooling;
    static size_t maxPoolingWidth;
    static size_t maxPoolingHeight;

    static size_t kollOfFilters;

    static size_t filterWidth;
    static size_t filterHeight;
};
#endif //ARTIFICIALNN_CONVOLUTIONNETWORK_H
