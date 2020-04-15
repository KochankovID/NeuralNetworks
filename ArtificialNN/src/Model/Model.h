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

        Tensor<T> predict(const Tensor<T>& x);
        Matrix<Tensor<T>> predict(const Matrix<Tensor<T>>& x);


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
                const std::vector<Metr<T>*>& metrixes, const Matrix<T>& metrix_t, size_t base) const;

        void changingWeight(const Matrix<Matrix<T>>& error, const ImpulsGrad<T>& G);

        void errorCalculate(Matrix<Matrix<T>>& error, size_t koll_of_examples, size_t batch_size,  size_t bt);

        void showProgress(size_t koll_of_examples, size_t num_of_examples) const;

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
        Matrix<Matrix<T>> error(1, batch_size);
        Matrix<T> metrix_t(metrixes.size(), koll_of_examples);

        cout << "Number of training examples: " << koll_of_examples << endl;

        for(size_t ep = 0; ep < epoches; ep++){
            cout << endl << "epoch: " << ep+1 << '/' << epoches << endl;
            for (size_t bt = 0; bt < koll_of_examples/batch_size; bt++) {

                    for (size_t ex = 0; ex < batch_size; ex++) {
                        predict(train_data[0][bt * batch_size + ex]);

                        error[0][ex] = ANN::loss_function(loss_func_der, TENSOR_OUT.back()[0],
                                                          train_out[0][bt * batch_size + ex][0]);

                        for (size_t i = 0; i < metrixes.size(); i++) {
                            metrix_t[i][bt * batch_size + ex] = metric_function(*(metrixes[i]), TENSOR_OUT.back()[0],
                                                                                train_out[0][bt * batch_size + ex][0]);
                        }
                    }
                cout << std::setw(8) << std::right << bt*batch_size << '/' << koll_of_examples << " ";
                showProgress(koll_of_examples, bt*batch_size);
                errorCalculate(error, koll_of_examples, batch_size, bt);
                changingWeight(error, G);
                showMetrix(ep, bt, koll_of_examples, batch_size, metrixes, metrix_t, bt*batch_size);
            }

            if((koll_of_examples%batch_size != 0)){
                size_t base = (koll_of_examples-(koll_of_examples % batch_size));
                for(size_t ex = 0; ex < koll_of_examples % batch_size; ex++){
                    predict(train_data[0][base+ex]);

                    error[0][ex] = ANN::loss_function(loss_func_der, TENSOR_OUT.back()[0],
                            train_out[0][base+ex][0]);

                    for(size_t i = 0; i < metrixes.size();i++){
                        metrix_t[i][base+ex] = metric_function(*(metrixes[i]), TENSOR_OUT.back()[0],
                                                                        train_out[0][base+ex][0]);
                    }
                }

                cout << std::setw(8) << std::right << base << '/' << koll_of_examples << " ";
                showProgress(koll_of_examples, base);
                errorCalculate(error, koll_of_examples, koll_of_examples % batch_size, koll_of_examples/batch_size);
                changingWeight(error, G);
                showMetrix(ep, koll_of_examples/batch_size, koll_of_examples, koll_of_examples % batch_size,
                        metrixes, metrix_t, base);
            }
        }
    }

    template<typename T>
    Tensor<T> Model<T>::predict(const Tensor<T>& x) {
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
    Matrix<Tensor<T>> Model<T>::predict(const Matrix<Tensor<T>>& x) {
        Matrix<Tensor<T>> result(1, x.getM());
        for(size_t i = 0; i < x.getM(); i++){
            result[0][i] = predict(x[0][i]);
        }
        return result;
    }

    template<typename T>
    void Model<T>::showMetrix(size_t ep, size_t bt, size_t koll_of_examples, size_t batch_size,
                              const std::vector<Metr<T> *> &metrixes, const Matrix<T> &metrix_t, size_t base) const {
        if(ep == 0) {
            for(size_t i = 0; i < metrixes.size(); i++) {
                cout << " " << metrixes[i]->getName() << ": " << std::setw(6) << std::setprecision(3)
                    << std::left << std::setfill('0') << mean(metrix_t[i], base + batch_size);
            }
        }else{
            for(size_t i = 0; i < metrixes.size(); i++) {
                cout << " " << metrixes[i]->getName() << ": " << std::setw(6) << std::setprecision(3)
                     << std::left << std::setfill('0') << mean(metrix_t[i], koll_of_examples);
            }
        }
        cout << std::setfill(' ');
        cout << std::endl;
    }

    template<typename T>
    void Model<T>::changingWeight(const Matrix<Matrix<T>> &error, const ImpulsGrad<T>& G) {
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
    }

    template<typename T>
    void
    Model<T>::errorCalculate(Matrix<Matrix<T>> &error, size_t koll_of_examples, size_t batch_size, size_t bt) {
        for (size_t i = 1; i < batch_size; i++) {
            error[0][0] += error[0][i];
        }
        for (size_t i = 0; i < error[0][0].getM(); i++) {
            error[0][0][0][i] /= batch_size;
        }
    }

    template<typename T>
    void Model<T>::showProgress(size_t koll_of_examples, size_t num_of_examples) const {
        double percents = 30 * ((num_of_examples / (koll_of_examples / 100)) * 0.01);
        std::string out = "[";
        for(int i = 0; i < 30; i++){
            if(i == round(percents)){
                out += ">";
            }else{
                if(i < percents) {
                    out += "=";
                }else{
                    out += "-";
                }
            }
        }
        out += "]";
        cout << out;

    }
}

#endif //ARTIFICIALNN_MODEL_H
