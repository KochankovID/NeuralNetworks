#ifndef ARTIFICIALNN_MODEL_H
#define ARTIFICIALNN_MODEL_H

#include "ConvolutionLayers.h"
#include "DenceLayers.h"
#include "MaxpoolingLayers.h"
#include "FlattenLayers.h"
#include "vector"
#include "Metrix.h"

namespace ANN {

    using std::cout;
    using std::endl;

    template <typename T>
    class Model{
    public:
        Model();

        void add(const Layer<T>& layer);

        void learnModel(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out,
                size_t batch_size, size_t epoches, const Metr<T>& loss_func_der = RMS_errorD<T>(),
                const std::vector<Metr<T>>& metrixes = std::vector<Metr<T>>());

        ~Model() = default;

    private:
        std::vector<Layer<T>> arr_;
        std::vector<Tensor<T>> TENSOR_IN;
        std::vector<Tensor<T>> TENSOR_OUT;
        std::vector<Tensor<T>> TENSOR_IN_D;
        std::vector<Tensor<T>> TENSOR_OUT_D;
    };

    template<typename T>
    Model<T>::Model() {}

    template<typename T>
    void Model<T>::add(const Layer<T> &layer) {
        arr_.push_back(layer);
        TENSOR_IN.push_back(Tensor<T>());
        TENSOR_OUT.push_back(Tensor<T>());
        TENSOR_IN_D.push_back(Tensor<T>());
        TENSOR_OUT_D.push_back(Tensor<T>());
    }

    template<typename T>
    void Model<T>::learnModel(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out, size_t batch_size,
            size_t epoches, const Metr<T> &loss_func_der, const std::vector<Metr<T>> &metrixes) {

        if(train_data.getM() != train_out.getM()){
            throw std::runtime_error("Number of classes in train data and in validation are not equal!");
        }

        size_t koll_of_examples = train_data.getM();

        cout << "Number of trining examples: " << koll_of_examples;

        Matrix<T> error;
        Matrix<T> metrix_t(metrixes.size(), koll_of_examples/batch_size);

        for(size_t ep = 0; ep < epoches; ep++){
            for (size_t bt = 0; bt < koll_of_examples/batch_size; bt++) {
                if(bt == koll_of_examples/batch_size - 1){
                    for(size_t ex = 0; ex < koll_of_examples % batch_size; ex++){

                    }
                }else {
                    for (size_t ex = 0; ex < batch_size; ex++) {
                        for(size_t i = 0; i < arr_; i++){
                            if(i == 0) {
                                TENSOR_IN[i] = train_data[0][bt*batch_size+ex];
                                TENSOR_OUT[i] = arr_[i].passThrough(TENSOR_IN[i]);
                            }else{
                                TENSOR_IN[i] = TENSOR_OUT[i-1];
                                TENSOR_OUT[i] = arr_[i].passThrough(TENSOR_IN[i]);
                            }
                        }

                        error = loss_function(loss_func_der, TENSOR_OUT.end(), train_out[bt*batch_size+ex]);
                        for(size_t i = 0; i < metrixes.size();i++){
                            metrix_t[i] = metric_function(metrixes[i], TENSOR_OUT.end(), train_out[bt*batch_size+ex]);
                        }

                        for(size_t i = arr_.size()-1; i >=0; i--){
                            if(i == arr_.size()-1){
                                TENSOR_IN_D[i] = error;
                                TENSOR_OUT_D[i] = arr_[i].
                            }else{

                            }
                        }
                    }
                }
            }
        }



    }
}

#endif //ARTIFICIALNN_MODEL_H
