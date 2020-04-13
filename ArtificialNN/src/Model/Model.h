#ifndef ARTIFICIALNN_MODEL_H
#define ARTIFICIALNN_MODEL_H

#include "ConvolutionLayers.h"
#include "DenceLayers.h"
#include "MaxpoolingLayers.h"
#include "FlattenLayers.h"
#include "vector"
#include "LearnNeyron.h"
#include "Metrix.h"

namespace ANN {

    using std::cout;
    using std::endl;

    template <typename T>
    class Model{
    public:
        Model();

        void add(Layer<T>* layer);

        void learnModel(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out,
                size_t batch_size, size_t epoches, const ImpulsGrad<T>& G = SGD<T>(1), const Metr<T>& loss_func_der = RMS_errorD<T>(),
                const std::vector<Metr<T>*>& metrixes = std::vector<Metr<T>*>());

        Tensor<T> predict(Tensor<T> x);


        ~Model() = default;

    private:
        std::vector<Layer<T>*> arr_;
        std::vector<Tensor<T>> TENSOR_IN;
        std::vector<Tensor<T>> TENSOR_OUT;
        std::vector<Tensor<T>> TENSOR_IN_D;
        std::vector<Tensor<T>> TENSOR_OUT_D;

        double mean(const double* arr, size_t len) const {
            double result = std::accumulate(arr, arr+len, 0.0);
            return result/len;
        }

        void showMetrix(size_t ep, size_t bt, size_t koll_of_examples, size_t batch_size,
                        const std::vector<Metr<T>*>& metrixes, const Matrix<T>& metrix_t) const;
    };

    template<typename T>
    Model<T>::Model() {}

    template<typename T>
    void Model<T>::add(Layer<T>* layer) {
        arr_.push_back(layer);
        TENSOR_IN.push_back(Tensor<T>());
        TENSOR_OUT.push_back(Tensor<T>());
        TENSOR_IN_D.push_back(Tensor<T>());
        TENSOR_OUT_D.push_back(Tensor<T>());
    }

    template<typename T>
    void Model<T>::learnModel(Matrix<Tensor<T>> train_data, Matrix<Tensor<T>> train_out, size_t batch_size,
            size_t epoches, const ImpulsGrad<T>& G, const Metr<T> &loss_func_der, const std::vector<Metr<T>*> &metrixes) {

        if(train_data.getM() != train_out.getM()){
            throw std::runtime_error("Number of classes in train data and in validation are not equal!");
        }

        size_t koll_of_examples = train_data.getM();

        cout << "Number of trining examples: " << koll_of_examples << endl;

        Matrix<Matrix<T>> error(1, batch_size);

        Matrix<T> metrix_t(metrixes.size(), koll_of_examples);

        for(size_t ep = 0; ep < epoches; ep++){

            cout << endl << "epoch: " << ep+1 << '/' << epoches << endl;

            for (size_t bt = 0; bt < koll_of_examples/batch_size; bt++) {

                cout << std::setw(8) << std::right << bt*batch_size << '/' << koll_of_examples << " ";
                cout << "[";
                cout.flush();

                if((bt == koll_of_examples/batch_size - 1)&&(koll_of_examples%batch_size != 0)){
                    for(size_t ex = 0; ex < koll_of_examples % batch_size; ex++){
                        predict(train_data[0][bt*batch_size+ex]);

                        error[0][ex] = ANN::loss_function(loss_func_der, TENSOR_OUT.back()[0],
                                train_out[0][bt*batch_size+ex][0]);

                        for(size_t i = 0; i < metrixes.size();i++){
                            metrix_t[i][bt*batch_size+ex] = metric_function(*(metrixes[i]), TENSOR_OUT.back()[0],
                                    train_out[0][bt*batch_size+ex][0]);
                        }

                        cout << "||";
                        cout.flush();
                    }
                }else {
                    for (size_t ex = 0; ex < batch_size; ex++) {
                        predict(train_data[0][bt * batch_size + ex]);

                        error[0][ex] = ANN::loss_function(loss_func_der, TENSOR_OUT.back()[0],
                                                          train_out[0][bt * batch_size + ex][0]);

                        for (size_t i = 0; i < metrixes.size(); i++) {
                            metrix_t[i][bt * batch_size + ex] = metric_function(*(metrixes[i]),
                                                                                TENSOR_OUT.back()[0],
                                                                                train_out[0][bt * batch_size + ex][0]);
                        }
                        cout << "||";
                        cout.flush();
                    }
                }

                if((bt == koll_of_examples/batch_size - 1)&&(koll_of_examples%batch_size != 0)){
                    for(size_t i = 1; i < koll_of_examples % batch_size; i++){
                        error[0][0] += error[0][i];
                    }
                    for(size_t i = 0; i < error[0][0].getM(); i++){
                        error[0][0][0][i] /= koll_of_examples % batch_size;
                    }
                }else {
                    for (size_t i = 1; i < batch_size; i++) {
                        error[0][0] += error[0][i];
                    }
                    for (size_t i = 0; i < error[0][0].getM(); i++) {
                        error[0][0][0][i] /= batch_size;
                    }
                }
                for(int i = arr_.size()-1; i >=0; i--){
                    if(i == arr_.size()-1){
                        TENSOR_IN_D[i] = error[0][0];
                        TENSOR_OUT_D[i] = arr_[i]->BackPropagation(TENSOR_IN_D[i], TENSOR_IN[i]);
                    }else{
                        TENSOR_IN_D[i] = TENSOR_OUT_D[i+1];
                        TENSOR_OUT_D[i] = arr_[i]->BackPropagation(TENSOR_IN_D[i], TENSOR_IN[i]);
                    }
                    arr_[i]->GradDes(G,TENSOR_IN[i]);
                }

                showMetrix(ep, bt, koll_of_examples, batch_size, metrixes, metrix_t);

                cout << endl;
            }
        }
    }

    template<typename T>
    Tensor<T> Model<T>::predict(Tensor<T> x) {
        for(size_t i = 0; i < arr_.size(); i++){
            if(i == 0) {
                TENSOR_IN[i] = x;
                TENSOR_OUT[i] = arr_[i]->passThrough(TENSOR_IN[i]);
            }else{
                TENSOR_IN[i] = TENSOR_OUT[i-1];
                TENSOR_OUT[i] = arr_[i]->passThrough(TENSOR_IN[i]);
            }
        }
        return TENSOR_OUT.back();
    }

    template<typename T>
    void Model<T>::showMetrix(size_t ep, size_t bt, size_t koll_of_examples, size_t batch_size,
                              const std::vector<Metr<T> *> &metrixes, const Matrix<T> &metrix_t) const {
        if(ep == 0) {
            cout << "]";
            for(size_t i = 0; i < metrixes.size(); i++) {
                if(bt == koll_of_examples/batch_size - 1){
                    cout << " " << metrixes[i]->getName() << ": " << std::setw(6) << std::setprecision(3)
                         << std::left << std::setfill('0') << mean(metrix_t[i], bt * batch_size + koll_of_examples % batch_size);
                }
                else {
                    cout << " " << metrixes[i]->getName() << ": " << std::setw(6) << std::setprecision(3)
                        << std::left << std::setfill('0') << mean(metrix_t[i], bt * batch_size + batch_size);
                }
            }
        }else{
            cout << "]";
            for(size_t i = 0; i < metrixes.size(); i++) {
                cout << " " << metrixes[i]->getName() << ": " << std::setw(6) << std::setprecision(3)
                     << std::left << std::setfill('0') << mean(metrix_t[i], koll_of_examples);
            }
        }
        cout << std::setfill(' ');
    }
}

#endif //ARTIFICIALNN_MODEL_H
