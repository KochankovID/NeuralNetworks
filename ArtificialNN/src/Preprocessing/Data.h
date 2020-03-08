#ifndef ARTIFICIALNN_DATA_H
#define ARTIFICIALNN_DATA_H

#include <vector>
#include <string>
#include <fstream>
#include "Neyrons.h"

template <typename T>
void getDataFromTextFile(std::vector<T>& input, const std::string& str);

template <typename T>
void saveWeightsTextFile(ANN::Neyron<T>& neyron, const std::string& str);

template <typename T>
void getWeightsTextFile(ANN::Neyron<T>& neyron, const std::string& str);

template <typename T>
void getDataFromTextFile(std::vector<T>& input, const std::string& str){
    std::ifstream TeachChoose;
    TeachChoose.open(str);
    for (int i = 0; i < input.size(); i++) {
        TeachChoose >> input[i];
    }
    TeachChoose.close();
}

template <typename T>
void saveWeightsTextFile(ANN::Neyron<T>& neyron, const std::string& str){
    std::ofstream fWeights;
    fWeights.open(str);
    fWeights << neyron;
    fWeights.close();
}

template <typename T>
void getWeightsTextFile(ANN::Neyron<T>& neyron, const std::string& str){
    std::ifstream fWeights;
    fWeights.open(str);
    fWeights >> neyron;
    fWeights.close();
}

#endif //ARTIFICIALNN_DATA_H
