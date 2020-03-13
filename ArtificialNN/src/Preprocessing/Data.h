#ifndef ARTIFICIALNN_DATA_H
#define ARTIFICIALNN_DATA_H

#include <vector>
#include <string>
#include <fstream>
#include "Neyrons.h"

namespace ANN {
    template<typename T>
    void getDataFromTextFile(ANN::Matrix<ANN::Matrix<T>> &input, const std::string &str);

    template<typename T>
    void saveWeightsTextFile(ANN::Neyron<T> &neyron, const std::string &str);

    template<typename T>
    void saveWeightsTextFile(ANN::Matrix<ANN::Neyron<T>> &neyron, const std::string &str);

    template<typename T>
    void getWeightsTextFile(ANN::Neyron<T> &neyron, const std::string &str);

    template<typename T>
    void getWeightsTextFile(ANN::Matrix<ANN::Neyron<T> > &neyron, const std::string &str);

// Класс исключения ------------------------------------------------------
    class DataExeption : public std::logic_error {
    public:
        DataExeption(std::string str) : std::logic_error(str) {};

        ~DataExeption() {};
    };

    template<typename T>
    void getDataFromTextFile(ANN::Matrix<ANN::Matrix<T> > &input, const std::string &str) {
        std::ifstream TeachChoose;
        TeachChoose.open(str);
        if (!TeachChoose.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        for (int i = 0; i < input.getN(); i++) {
            for (int j = 0; j < input.getM(); j++) {
                TeachChoose >> input[i][j];
            }
        }
        TeachChoose.close();
    }

    template<typename T>
    void saveWeightsTextFile(ANN::Neyron<T> &neyron, const std::string &str) {
        std::ofstream fWeights;
        fWeights.open(str);
        if (!fWeights.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        fWeights << neyron;
        fWeights.close();
    }

    template<typename T>
    void saveWeightsTextFile(ANN::Matrix<ANN::Neyron<T>> &neyron, const std::string &str) {
        std::ofstream fWeights;
        fWeights.open(str);
        if (!fWeights.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        for (int i = 0; i < neyron.getN(); i++) {
            for (int j = 0; j < neyron.getM(); j++) {
                fWeights << neyron[i][j];
            }
        }
        fWeights.close();
    }

    template<typename T>
    void getWeightsTextFile(ANN::Neyron<T> &neyron, const std::string &str) {
        std::ifstream fWeights;
        fWeights.open(str);
        if (!fWeights.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        fWeights >> neyron;
        fWeights.close();
    }

    template<typename T>
    void getWeightsTextFile(ANN::Matrix<ANN::Neyron<T> > &neyron, const std::string &str) {
        std::ifstream fWeights;
        fWeights.open(str);
        if (!fWeights.is_open()) {
            throw DataExeption("Файл ненайден!");
        }
        for (int i = 0; i < neyron.getN(); i++) {
            for (int j = 0; j < neyron.getM(); j++) {
                fWeights >> neyron[i][j];
            }
        }
        fWeights.close();
    }
}
#endif //ARTIFICIALNN_DATA_H
