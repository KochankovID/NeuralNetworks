#ifndef ARTIFICIALNN_MODEL_H
#define ARTIFICIALNN_MODEL_H

#include "ConvolutionLayers.h"
#include "DenceLayers.h"
#include "MaxpoolingLayers.h"
#include "FlattenLayers.h"
#include "vector"
#include "Metrix.h"

namespace ANN {

    template <typename T>
    class Model{
    public:
        Model();

        void add(const Layer<T>& layer);

        void learnModel(Matrix<Matrix<T>> train_data, Matrix<Matrix<T>> train_out,
                const Metr<T>& loss_func_der = RMS_errorD<T>(),
                        const std::vector<Metr<T>>& metrixes = std::vector<Metr<T>>());

        ~Model() = default;

    private:
        std::vector<Layer<T>> arr_;
        std::vector<Metr<T>> metrixes_;
        std::vector<Matrix<T>> MATRIX_IN;
        std::vector<Matrix<T>> MATRIX_OUT;
        std::vector<Tensor<T>> TENSOR_IN;
        std::vector<Tensor<T>> TENSOR_OUT;
        Metr<T> loss_func_der_;
    };

    template<typename T>
    Model<T>::Model() {}

    template<typename T>
    void Model<T>::add(const Layer<T> &layer) {
        arr_.push_back(layer);
        switch (layer.getType()){
            case "DenceLayer" :
                MATRIX_IN.push_back(Matrix<T>(1, dynamic_cast<DenceLayer<T>>(layer).getNumberImput()));
                MATRIX_OUT.push_back(Matrix<T>(1, dynamic_cast<DenceLayer<T>>(layer).getNumberImput()));
                break;
            case "FlattenLayer" :
                break;
            case "ConvolutionLayer" :
                break;
            case "MaxpoolingLayer" :
                break;
        }
    }

    template<typename T>
    void Model<T>::learnModel(Matrix<Matrix<T>> train_data, Matrix<Matrix<T>> train_out, const Metr<T> &loss_func_der,
                              const std::vector<Metr<T>> &metrixes) {
        loss_func_der_ = loss_func_der;
        for(auto metix : metrixes ){
            metrixes_.push_back(metix);
        }

    }
}

#endif //ARTIFICIALNN_MODEL_H
